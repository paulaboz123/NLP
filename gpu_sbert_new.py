# Uruchom tę komórkę tylko raz, aby zainstalować niezbędne biblioteki.
!pip install -q azure-ai-ml azure-identity pandas torch sentence-transformers scikit-learn

import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import DefaultAzureCredential
import pandas as pd

print("✅ Biblioteki zostały zainstalowane i zaimportowane.")

# --- KONFIGURACJA (WYPEŁNIJ SWOIMI DANYMI) ---

# Szczegóły Twojego Azure ML Workspace
SUBSCRIPTION_ID = "<TWOJ_SUBSCRIPTION_ID>"
RESOURCE_GROUP = "<TWOJA_RESOURCE_GROUP>"
WORKSPACE_NAME = "<TWOJA_NAZWA_WORKSPACE>"

# Nazwa klastra obliczeniowego z GPU, który zostanie stworzony w Azure
GPU_COMPUTE_NAME = "gpu-cluster-nc6"

# Ścieżka do Twojego lokalnego pliku z danymi
LOCAL_DATA_PATH = "your_dataset.csv"

# Nazwa dla pliku wynikowego, który zostanie pobrany
LOCAL_OUTPUT_PATH = "labeled_dataset_from_azure.csv"

# Próg pewności (confidence)
CONFIDENCE_THRESHOLD = 0.6

# Stworzenie podfolderu na kod, który zostanie wysłany do Azure
os.makedirs("azure_code", exist_ok=True)

# Definicja zawartości skryptu Python.
# Ten skrypt przyjmuje dane wejściowe z folderu 'inputs' i zapisuje wynik do folderu 'outputs'
# na zdalnej maszynie.
script_content = """
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import argparse
import os
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    # Dane wejściowe i wyjściowe są teraz prostymi ścieżkami do folderów
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    args = parser.parse_args()

    # Znajdź plik csv w folderze wejściowym (Azure przesyła go z losową nazwą)
    input_file = [f for f in os.listdir(args.input_folder) if f.endswith('.csv')][0]
    input_path = os.path.join(args.input_folder, input_file)
    
    print(f"Wczytywanie danych z: {input_path}")
    df = pd.read_csv(input_path)
    df["RELEVANT"] = df["RELEVANT"].astype(int)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"Model wczytany na: {device}")

    print("Tworzenie prototypów...")
    labeled_df = df.dropna(subset=["DEMAND_ID"]).copy()
    labeled_texts = labeled_df["TEXT"].astype(str).tolist()
    labeled_embeddings = model.encode(labeled_texts, batch_size=256, show_progress_bar=True)

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
    print(f"Kodowanie {len(unlabeled_texts)} zdań...")
    unlabeled_embeddings = model.encode(unlabeled_texts, batch_size=256, show_progress_bar=True)

    print("Wyszukiwanie najbliższych etykiet...")
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
    # Nazwa pliku wyjściowego jest stała
    output_path = os.path.join(args.output_folder, "labeled_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Wyniki zapisane w: {output_path}")

if __name__ == "__main__":
    main()
"""
with open("azure_code/label_and_correct.py", "w") as f:
    f.write(script_content)

# Definicja środowiska YAML
yaml_content = """
name: sbert-gpu-env-simple
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pytorch
  - pytorch-cuda=11.8
  - pip:
    - pandas==2.2.2
    - scikit-learn==1.5.0
    - sentence-transformers==3.0.1
"""
with open("azure_code/environment.yml", "w") as f:
    f.write(yaml_content)

print("✅ Pliki skryptu i środowiska zostały stworzone w folderze 'azure_code'.")

from azure.ai.ml import dsl
from azure.ai.ml.entities import Input, Output
import time

# Połączenie z Azure
credential = DefaultAzureCredential()
ml_client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
print(f"Połączono z workspace: {ml_client.workspace_name}")

# --- Krok 1: Stworzenie (lub pobranie) klastra GPU ---
try:
    compute = ml_client.compute.get(GPU_COMPUTE_NAME)
    print(f"Klaster obliczeniowy '{GPU_COMPUTE_NAME}' już istnieje.")
except Exception:
    print(f"Tworzenie nowego klastra obliczeniowego '{GPU_COMPUTE_NAME}'...")
    compute = AmlCompute(
        name=GPU_COMPUTE_NAME,
        size="Standard_NC6",
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=180,
    )
    ml_client.compute.begin_create_or_update(compute).result()
    print("Klaster obliczeniowy utworzony.")

# --- Krok 2: Stworzenie (lub pobranie) środowiska Python ---
env_name = "sbert-gpu-environment-simple-v1"
try:
    env = ml_client.environments.get(name=env_name, label="latest")
    print(f"Środowisko '{env_name}' już istnieje.")
except Exception:
    print(f"Tworzenie nowego środowiska '{env_name}'...")
    env = Environment(
        name=env_name,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="azure_code/environment.yml",
    )
    ml_client.environments.create_or_update(env)
    print("Środowisko utworzone.")
    env = ml_client.environments.get(name=env_name, label="latest")

# --- Krok 3: Definicja zadania (Job) ---
job = command(
    # Przesyła cały folder 'azure_code' i plik z danymi
    code="./azure_code",
    inputs={"data": Input(type="uri_file", path=LOCAL_DATA_PATH)},
    outputs={"results": Output(type="uri_folder")},
    command=(
        "python label_and_correct.py "
        # Specjalne zmienne Azure ML wskazujące na foldery na zdalnej maszynie
        "--input_folder ${{inputs.data}} "
        "--output_folder ${{outputs.results}} "
        f"--confidence_threshold {CONFIDENCE_THRESHOLD}"
    ),
    environment=f"{env.name}@latest",
    compute=GPU_COMPUTE_NAME,
    display_name="local_data_gpu_processing_job",
    experiment_name="local-data-experiments"
)

# --- Krok 4: Uruchomienie zadania i oczekiwanie na wyniki ---
print("\nUruchamianie zadania w Azure ML. To może potrwać od 15 do 45 minut w zależności od wielkości danych...")
returned_job = ml_client.jobs.create_or_update(job)
print(f"Zadanie uruchomione. Link do panelu Azure ML: {returned_job.studio_url}")

# Pętla sprawdzająca status zadania
while returned_job.status not in ["Completed", "Failed", "Canceled"]:
    print(f"Status zadania: {returned_job.status}... Oczekiwanie 60 sekund.")
    time.sleep(60)
    returned_job = ml_client.jobs.get(name=returned_job.name)

# --- Krok 5: Pobranie wyników ---
if returned_job.status == "Completed":
    print("\n✅ Zadanie zakończone pomyślnie! Pobieranie wyników...")
    
    # Ścieżka do folderu, gdzie zostaną pobrane wyniki
    download_dir = "azure_downloaded_results"
    ml_client.jobs.download(name=returned_job.name, download_path=download_dir, output_name="results")
    
    # Przeniesienie pliku do głównego folderu i zmiana nazwy
    source_file = os.path.join(download_dir, "named-outputs", "results", "labeled_dataset.csv")
    os.rename(source_file, LOCAL_OUTPUT_PATH)
    
    # Opcjonalne: usunięcie tymczasowego folderu pobierania
    import shutil
    shutil.rmtree(download_dir)
    
    print(f"\n✅✅✅ Wszystko gotowe! Wynikowy plik został zapisany lokalnie jako: '{LOCAL_OUTPUT_PATH}'")
    
    # Wyświetlenie kilku pierwszych wierszy pobranego pliku
    results_df = pd.read_csv(LOCAL_OUTPUT_PATH)
    print("\nPodgląd przetworzonych danych:")
    print(results_df.head())
    
else:
    print(f"\n❌ Zadanie zakończyło się niepowodzeniem ze statusem: {returned_job.status}. Sprawdź logi w panelu Azure ML.")

