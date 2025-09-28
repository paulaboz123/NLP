import os
import shutil
import time
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import DefaultAzureCredential
import pandas as pd

# ======================================================================================
# GŁÓWNY BLOK KODU: ORKIESTRACJA ZADANIA OBLICZENIOWEGO NA GPU W AZURE
# ======================================================================================

print("--- Rozpoczynanie procesu zdalnych obliczeń na GPU w Azure ---")

# --- Krok 1: Programowe stworzenie plików dla zadania ---
# Tworzymy tymczasowy folder, który zostanie wysłany do Azure
code_folder = "azure_temp_code"
os.makedirs(code_folder, exist_ok=True)

# Definicja skryptu Python jako string. To jest kod, który FAKTYCZNIE wykona się na GPU.
script_content = """
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import argparse
import os
from collections import defaultdict

# Ten skrypt jest uruchamiany na zdalnej maszynie GPU w Azure
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--confidence_threshold", type=float)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.input_data, "data_to_process.csv"))
    df["RELEVANT"] = df["RELEVANT"].astype(int)
    
    device = 'cuda'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"Model wczytany na: {device}")

    labeled_df = df.dropna(subset=["DEMAND_ID"]).copy()
    labeled_texts = labeled_df["TEXT"].astype(str).tolist()
    labeled_embeddings = model.encode(labeled_texts, batch_size=512, show_progress_bar=True)

    label_embeddings_dict = defaultdict(list)
    for i in range(len(labeled_df)):
        label = labeled_df.iloc[i]["DEMAND_ID"]
        embedding = labeled_embeddings[i]
        label_embeddings_dict[label].append(embedding)

    prototype_embeddings_list = []
    prototype_labels = []
    for label, embeddings in label_embeddings_dict.items():
        prototype_embeddings_list.append(np.mean(embeddings, axis=0))
        prototype_labels.append(label)
    
    prototype_embeddings = torch.tensor(np.array(prototype_embeddings_list), device=device)

    unlabeled_df = df[df["DEMAND_ID"].isnull()].copy()
    unlabeled_texts = unlabeled_df["TEXT"].astype(str).tolist()
    print(f"Kodowanie {len(unlabeled_texts)} zdań na GPU...")
    unlabeled_embeddings = model.encode(unlabeled_texts, batch_size=512, show_progress_bar=True)

    hits = util.semantic_search(torch.tensor(unlabeled_embeddings).to(device), prototype_embeddings, top_k=1)
    
    update_indices = []
    new_labels = []
    unlabeled_df_indices = unlabeled_df.index

    for i, hit_list in enumerate(hits):
        if hit_list and hit_list[0]['score'] >= args.confidence_threshold:
            original_df_index = unlabeled_df_indices[i]
            update_indices.append(original_df_index)
            new_labels.append(prototype_labels[hit_list[0]['corpus_id']])

    if update_indices:
        df.loc[update_indices, "DEMAND_ID"] = new_labels
        df.loc[update_indices, "RELEVANT"] = 1

    os.makedirs(args.output_folder, exist_ok=True)
    df.to_csv(os.path.join(args.output_folder, "labeled_dataset.csv"), index=False)
    print(f"Wyniki zapisane.")
"""
with open(os.path.join(code_folder, "label_script.py"), "w") as f:
    f.write(script_content)

# Definicja środowiska Conda jako string
yaml_content = """
name: sbert-gpu-env
channels:
  - pytorch
  - nvidia
dependencies:
  - python=3.11
  - pip
  - pytorch
  - pytorch-cuda=11.8
  - pip:
    - pandas==2.2.2
    - scikit-learn==1.5.0
    - sentence-transformers==3.0.1
    - azureml-mlflow # Potrzebne do działania w środowisku Azure
"""
with open(os.path.join(code_folder, "environment.yml"), "w") as f:
    f.write(yaml_content)

print("✅ Tymczasowe pliki skryptu i środowiska zostały stworzone.")


# --- Krok 2: Zapisanie lokalnego DataFrame do pliku tymczasowego ---
# Azure ML Job potrzebuje pliku jako wejścia, a nie obiektu w pamięci
temp_input_folder = "azure_temp_input"
os.makedirs(temp_input_folder, exist_ok=True)
df.to_csv(os.path.join(temp_input_folder, "data_to_process.csv"), index=False)
print("✅ DataFrame zapisany do tymczasowego pliku CSV.")


# --- Krok 3: Połączenie z Azure i definicja zasobów ---
credential = DefaultAzureCredential()
ml_client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
print(f"\nPołączono z workspace: {ml_client.workspace_name}")

# Definicja klastra GPU
gpu_compute_name = "gpu-cluster-nc6"
try:
    compute = ml_client.compute.get(gpu_compute_name)
    print(f"Klaster obliczeniowy '{gpu_compute_name}' już istnieje.")
except Exception:
    print(f"Tworzenie nowego klastra obliczeniowego '{gpu_compute_name}'...")
    compute = AmlCompute(name=gpu_compute_name, size="Standard_NC6", min_instances=0, max_instances=2)
    ml_client.compute.begin_create_or_update(compute).result()
    print("Klaster utworzony.")

# Definicja środowiska
env_name = "sbert-gpu-job-env"
try:
    env = ml_client.environments.get(name=env_name, label="latest")
    print(f"Środowisko '{env_name}' już istnieje.")
except Exception:
    print(f"Tworzenie nowego środowiska '{env_name}'...")
    env = Environment(name=env_name, image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04", conda_file=os.path.join(code_folder, "environment.yml"))
    ml_client.environments.create_or_update(env)
    print("Środowisko utworzone.")
    env = ml_client.environments.get(name=env_name, label="latest")


# --- Krok 4: Definicja i uruchomienie komponentu (Azure ML Command Job) ---
job = command(
    code=code_folder,
    command=(
        "python label_script.py "
        "--input_data ${{inputs.data_folder}} "
        "--output_folder ${{outputs.results_folder}} "
        f"--confidence_threshold {CONFIDENCE_THRESHOLD}"
    ),
    inputs={"data_folder": Input(type="uri_folder", path=temp_input_folder)},
    outputs={"results_folder": Output(type="uri_folder")},
    environment=f"{env.name}@latest",
    compute=gpu_compute_name,
    display_name="notebook_gpu_labeling_job",
    experiment_name="interactive-notebook-runs"
)

print("\n🚀 Uruchamianie zadania na klastrze GPU w Azure. To może potrwać 15-45 minut.")
returned_job = ml_client.jobs.create_or_update(job)
print(f"🔗 Zadanie uruchomione. Możesz je śledzić w Azure ML Studio: {returned_job.studio_url}")


# --- Krok 5: Oczekiwanie na zakończenie i pobranie wyników ---
# Ta pętla będzie czekać, aż zadanie w Azure się zakończy
ml_client.jobs.stream(returned_job.name) # Pokazuje logi na żywo

# Po zakończeniu pobieramy wyniki
download_dir = "azure_downloaded_results"
ml_client.jobs.download(name=returned_job.name, download_path=download_dir, output_name="results_folder")

# Wczytanie pobranych wyników z powrotem do DataFrame w notatniku
results_path = os.path.join(download_dir, "named-outputs", "results_folder", "labeled_dataset.csv")
results_df = pd.read_csv(results_path)

print(f"\n\n✅✅✅ Proces zakończony! Wyniki zostały pobrane i wczytane do DataFrame 'results_df'.")


# --- Krok 6: Sprzątanie ---
# Usunięcie tymczasowych folderów
shutil.rmtree(code_folder)
shutil.rmtree(temp_input_folder)
shutil.rmtree(download_dir)
print("✅ Usunięto tymczasowe pliki i foldery.")


# --- Wyświetlenie końcowego wyniku w notatniku ---
print("\nPodgląd przetworzonych danych:")
print(results_df.head())

print("\nNowy rozkład wartości w kolumnie 'RELEVANT':")
print(results_df[RELEVANT_COLUMN].value_counts())
