# Uruchom tę komórkę tylko raz, aby zainstalować niezbędne biblioteki.
!pip install -q azure-ai-ml azure-identity pandas torch sentence-transformers scikit-learn

print("✅ Biblioteki zostały zainstalowane.")

# --- KONFIGURACJA (WYPEŁNIJ SWOIMI DANYMI) ---

# Szczegóły Twojego Azure ML Workspace
SUBSCRIPTION_ID = "<TWOJ_SUBSCRIPTION_ID>"
RESOURCE_GROUP = "<TWOJA_RESOURCE_GROUP>"
WORKSPACE_NAME = "<TWOJA_NAZWA_WORKSPACE>"

# Nazwa klastra obliczeniowego z GPU, który zostanie stworzony w Azure
# Możesz zostawić domyślną nazwę.
GPU_COMPUTE_NAME = "gpu-cluster-nc6"

# Ścieżka do Twojego lokalnego pliku z danymi
# Plik musi znajdować się w tym samym folderze, co ten notatnik.
LOCAL_DATA_PATH = "your_dataset.csv"

# Próg pewności (confidence) do przypisania etykiety i zmiany flagi RELEVANT
# Wartość od 0.0 do 1.0. Dobry punkt startowy to 0.6.
CONFIDENCE_THRESHOLD = 0.6

import os

# Stworzenie podfolderu na kod, który zostanie wysłany do Azure
os.makedirs("azure_code", exist_ok=True)

# Definicja zawartości skryptu Python jako string
script_content = """
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import argparse
import os
import mlflow
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True, help="Ścieżka do danych wejściowych")
    parser.add_argument("--output_data", type=str, required=True, help="Ścieżka do zapisu wyników")
    parser.add_argument("--confidence_threshold", type=float, default=0.6, help="Próg pewności")
    args = parser.parse_args()

    mlflow.start_run()
    print(f"Rozpoczęto śledzenie z MLflow dla uruchomienia: {mlflow.active_run().info.run_id}")
    mlflow.log_param("confidence_threshold", args.confidence_threshold)

    print(f"Wczytywanie danych z: {args.input_data}")
    df = pd.read_csv(args.input_data)
    df["RELEVANT"] = df["RELEVANT"].astype(int)
    
    initial_null_labels = int(df["DEMAND_ID"].isnull().sum())
    mlflow.log_metric("initial_null_labels", initial_null_labels)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'all-MiniLM-L6-v2'
    print(f"Wczytywanie modelu SBERT '{model_name}' na urządzenie: '{device}'")
    model = SentenceTransformer(model_name, device=device)

    print("Tworzenie semantycznych prototypów...")
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
    print(f"Kodowanie {len(unlabeled_texts)} nieoznaczonych zdań na GPU...")
    unlabeled_embeddings = model.encode(unlabeled_texts, batch_size=256, show_progress_bar=True)

    print("Wyszukiwanie najbliższych etykiet...")
    hits = util.semantic_search(unlabeled_embeddings, torch.tensor(unlabeled_embeddings).to(device), top_k=1)
    
    update_indices = []
    new_labels = []
    unlabeled_df_indices = unlabeled_df.index

    for i, hit_list in enumerate(hits):
        if hit_list and hit_list[0]['score'] >= args.confidence_threshold:
            original_df_index = unlabeled_df_indices[i]
            update_indices.append(original_df_index)
            new_labels.append(prototype_labels[hit_list[0]['corpus_id']])

    if update_indices:
        flipped_relevance_count = (df.loc[update_indices, "RELEVANT"] == 0).sum()
        df.loc[update_indices, "DEMAND_ID"] = new_labels
        df.loc[update_indices, "RELEVANT"] = 1
        mlflow.log_metric("labels_assigned", len(update_indices))
        mlflow.log_metric("relevance_flipped_count", flipped_relevance_count)
    else:
        mlflow.log_metric("labels_assigned", 0)
        mlflow.log_metric("relevance_flipped_count", 0)

    final_null_labels = int(df["DEMAND_ID"].isnull().sum())
    mlflow.log_metric("final_null_labels", final_null_labels)

    os.makedirs(args.output_data, exist_ok=True)
    output_path = os.path.join(args.output_data, "labeled_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Wyniki zapisane w: {output_path}")

    mlflow.end_run()

if __name__ == "__main__":
    main()
"""
with open("azure_code/label_and_correct.py", "w") as f:
    f.write(script_content)

# Definicja zawartości pliku środowiska YAML jako string
yaml_content = """
name: sbert-gpu-env
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - pip:
    - pandas==2.2.2
    - scikit-learn==1.5.0
    - sentence-transformers==3.0.1
    - mlflow
    - azureml-mlflow
    - datasets
    - argparse
"""
with open("azure_code/environment.yml", "w") as f:
    f.write(yaml_content)

print("✅ Pliki 'label_and_correct.py' oraz 'environment.yml' zostały stworzone w folderze 'azure_code'.")

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Data, Environment, AmlCompute
from azure.identity import DefaultAzureCredential

# Uwierzytelnienie - otworzy się okno przeglądarki do zalogowania
credential = DefaultAzureCredential()
ml_client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
print(f"Połączono z workspace: {ml_client.workspace_name}")

# --- Klaster Obliczeniowy (GPU) ---
try:
    compute = ml_client.compute.get(GPU_COMPUTE_NAME)
    print(f"Klaster obliczeniowy '{GPU_COMPUTE_NAME}' już istnieje.")
except Exception:
    print(f"Tworzenie nowego klastra obliczeniowego '{GPU_COMPUTE_NAME}'...")
    # Standard_NC6 to popularny i relatywnie tani wybór z 1 GPU NVIDIA K80
    compute = AmlCompute(
        name=GPU_COMPUTE_NAME,
        size="Standard_NC6",
        min_instances=0, # Automatyczne skalowanie do 0, gdy nieużywany (oszczędność)
        max_instances=4,
        idle_time_before_scale_down=180,
    )
    ml_client.compute.begin_create_or_update(compute).result()
    print("Klaster obliczeniowy utworzony pomyślnie.")

# --- Środowisko Python ---
env_name = "sbert-gpu-environment-v2"
try:
    env = ml_client.environments.get(name=env_name, label="latest")
    print(f"Środowisko '{env_name}' już istnieje.")
except Exception:
    print(f"Tworzenie nowego środowiska '{env_name}'...")
    env = Environment(
        name=env_name,
        description="Environment with PyTorch, CUDA, and SBERT for GPU processing.",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="azure_code/environment.yml",
    )
    ml_client.environments.create_or_update(env)
    print("Środowisko utworzone pomyślnie.")
    env = ml_client.environments.get(name=env_name, label="latest")

# --- Dane Wejściowe ("Data Asset") ---
data_asset_name = "unlabeled-demands-data"
try:
    data_asset = ml_client.data.get(name=data_asset_name, label="latest")
    print(f"Data asset '{data_asset_name}' już istnieje.")
except Exception:
    print(f"Tworzenie nowego data asset '{data_asset_name}' z pliku {LOCAL_DATA_PATH}...")
    my_data = Data(
        path=LOCAL_DATA_PATH,
        type="uri_file",
        name=data_asset_name,
        description="Dataset for demand ID classification."
    )
    ml_client.data.create_or_update(my_data)
    print("Data asset utworzony pomyślnie.")
    data_asset = ml_client.data.get(name=data_asset_name, label="latest")

# --- Definicja i Uruchomienie Zadania (Job) ---
job_display_name = "fast-gpu-labeling-job"

job = command(
    code="./azure_code", # Folder z naszym skryptem i środowiskiem
    command=(
        "python label_and_correct.py "
        f"--input_data ${{inputs.data_to_label}} "
        f"--output_data ${{outputs.labeled_data}} "
        f"--confidence_threshold {CONFIDENCE_THRESHOLD}"
    ),
    inputs={"data_to_label": Input(type="uri_file", path=data_asset.id)},
    outputs={"labeled_data": Output(type="uri_folder")},
    environment=f"{env.name}@latest",
    compute=GPU_COMPUTE_NAME,
    display_name=job_display_name,
    experiment_name="demand-id-labeling-experiments"
)

print("\nUruchamianie zadania w Azure ML... To może potrwać chwilę.")
returned_job = ml_client.jobs.create_or_update(job)

print(f"\n✅ Zadanie uruchomione! Możesz śledzić jego postęp w Azure ML Studio:")
print(returned_job.studio_url)

# Uruchom tę komórkę, gdy zadanie się zakończy
job_name = returned_job.name # Pobiera nazwę zadania z poprzedniej komórki

# Ścieżka do folderu, gdzie zostaną pobrane wyniki
download_path = "azure_results"

print(f"Pobieranie wyników dla zadania '{job_name}' do folderu '{download_path}'...")
ml_client.jobs.download(name=job_name, download_path=download_path, output_name="labeled_data")

# Wynikowy plik CSV będzie się znajdował w: ./azure_results/named-outputs/labeled_data/labeled_dataset.csv
final_file_path = os.path.join(download_path, "named-outputs", "labeled_data", "labeled_dataset.csv")
print(f"\n✅ Wyniki pobrane! Przetworzony plik znajduje się w: {final_file_path}")

# Wyświetlenie kilku pierwszych wierszy przetworzonych danych
import pandas as pd
results_df = pd.read_csv(final_file_path)
results_df.head()
