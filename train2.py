# Użyj tej komendy, jeśli biblioteki nie są zainstalowane w Twoim środowisku
# %pip install pandas torch sentence-transformers scikit-learn

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import torch
import os
import random

# Ustawienia wyświetlania dla Pandas
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_rows', 100)

print("Biblioteki zostały zaimportowane.")
Use code with caution.
Python
Komórka 2: Konfiguracja (EDYTUJ TUTAJ)
Jedyne miejsce, które musisz edytować. Zawiera parametry dla wszystkich trzech kroków.
Generated python
# --- EDYTUJ TE WARTOŚCI ---

# Ścieżka do Twojego ORYGINALNEGO, surowego pliku CSV z danymi
INPUT_RAW_DATA_PATH = '/sciezka/do/twoich/oryginalnych/danych.csv'

# Ścieżka do LOKALNEGO folderu z modelem Sentence Transformer
MODEL_PATH = '/sciezka/do/twojego/modelu/lokalnego/'

# Ścieżka do zapisu JEDYNEGO, FINALNEGO pliku gotowego do treningu
OUTPUT_FINAL_TRAINING_DATA_PATH = './final_training_data_for_transformer.csv'

# Nazwy kolumn w Twoim pliku CSV
TEXT_COLUMN = 'text'
RELEVANT_COLUMN = 'relevant'
LABEL_COLUMN = 'label'

# --- PARAMETRY PROCESU (można dostosować) ---

# KROK 1: Oczyszczanie
SIMILARITY_THRESHOLD = 0.7 # Próg podobieństwa do automatycznej korekty etykiet

# KROK 2: Próbkowanie negatywów
NUM_CLUSTERS = 75           # Liczba grup tematycznych dla zdań negatywnych
NUM_SAMPLES_TO_SELECT = 5000 # Ile ostatecznie zdań negatywnych wybrać

# KROK 3: Montaż zbioru
HARD_NEGATIVES_RATIO = 1.0 # Stosunek "trudnych" negatywów do pozytywów (1.0 = tyle samo)

# --- Konfiguracja techniczna ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Konfiguracja zakończona.")
print(f"Używane urządzenie: {device.upper()}")
Use code with caution.
Python
Komórka 3: Wczytanie modelu i danych surowych
Generated python
print("--- Etap 0: Wczytywanie zasobów ---")

# Wczytanie modelu
print(f"Wczytywanie modelu z {MODEL_PATH}...")
model = SentenceTransformer(MODEL_PATH, device=device)
print("Model wczytany.")

# Wczytanie surowych danych
print(f"\nWczytywanie surowych danych z {INPUT_RAW_DATA_PATH}...")
df_raw = pd.read_csv(INPUT_RAW_DATA_PATH)
df_raw[LABEL_COLUMN] = df_raw[LABEL_COLUMN].fillna('brak_etykiety')

print("\nDane surowe wczytane pomyślnie. Rozkład etykiet 'relevant':")
print(df_raw[RELEVANT_COLUMN].value_counts())
Use code with caution.
Python
Komórka 4: Krok 1 - Oczyszczanie Danych
Ta komórka wykonuje całą logikę znajdowania i poprawiania błędnie oznaczonych danych. Jej wynikiem jest cleaned_df przekazywany do następnego kroku.
Generated python
print("\n--- KROK 1: OCZYSZCZANIE DANYCH ---")

positive_df_raw = df_raw[df_raw[RELEVANT_COLUMN] == 1].copy()
negative_df_raw = df_raw[df_raw[RELEVANT_COLUMN] == 0].copy()

if len(positive_df_raw) == 0:
    raise ValueError("Brak danych z 'relevant=1' do stworzenia centroidów.")

print("Obliczanie centroidów dla każdej etykiety...")
positive_embeddings = model.encode(positive_df_raw[TEXT_COLUMN].tolist(), show_progress_bar=True)
positive_df_raw['embedding'] = list(positive_embeddings)

label_centroids = {label: np.mean(np.vstack(group['embedding'].values), axis=0)
                   for label, group in positive_df_raw.groupby(LABEL_COLUMN)}
print(f"Obliczono {len(label_centroids)} centroidów.")

print("\nAnaliza danych 'relevant=0' w poszukiwaniu błędnych etykiet...")
negative_embeddings_raw = model.encode(negative_df_raw[TEXT_COLUMN].tolist(), show_progress_bar=True)
centroid_matrix = np.vstack(list(label_centroids.values()))
centroid_names = list(label_centroids.keys())
similarity_matrix = cosine_similarity(negative_embeddings_raw, centroid_matrix)

negative_df_raw['max_similarity'] = np.max(similarity_matrix, axis=1)
negative_df_raw['closest_label'] = [centroid_names[i] for i in np.argmax(similarity_matrix, axis=1)]

mislabeled_indices = negative_df_raw[negative_df_raw['max_similarity'] > SIMILARITY_THRESHOLD].index

cleaned_df = df_raw.copy()
cleaned_df.loc[mislabeled_indices, RELEVANT_COLUMN] = 1
cleaned_df.loc[mislabeled_indices, LABEL_COLUMN] = negative_df_raw.loc[mislabeled_indices, 'closest_label']

print(f"\nZnaleziono i poprawiono {len(mislabeled_indices)} błędnie oznaczonych wierszy.")
print("--- Krok 1 zakończony. ---")
Use code with caution.
Python
Komórka 5: Krok 2 - Próbkowanie zróżnicowanych negatywów
Ta komórka bierze cleaned_df z poprzedniego kroku, filtruje negatywy i wykonuje na nich próbkowanie. Wynikiem jest sampled_negatives_df przekazywany do ostatniego kroku.
Generated python
print("\n--- KROK 2: PRÓBKOWANIE ZRÓŻNICOWANYCH NEGATYWÓW ---")

negative_df_cleaned = cleaned_df[cleaned_df[RELEVANT_COLUMN] == 0].copy().reset_index(drop=True)

if len(negative_df_cleaned) < NUM_SAMPLES_TO_SELECT:
    raise ValueError(f"Liczba próbek do wyboru ({NUM_SAMPLES_TO_SELECT}) jest większa niż dostępnych zdań negatywnych ({len(negative_df_cleaned)}).")
print(f"Próbkowanie z {len(negative_df_cleaned)} oczyszczonych zdań negatywnych.")

print("Klastrowanie zdań negatywnych...")
negative_embeddings_cleaned = model.encode(negative_df_cleaned[TEXT_COLUMN].tolist(), show_progress_bar=True, batch_size=128)
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
negative_df_cleaned['cluster_id'] = kmeans.fit_predict(negative_embeddings_cleaned)

print("Wybór zróżnicowanych próbek (próbkowanie warstwowe)...")
weights = 1 / negative_df_cleaned.groupby('cluster_id')['cluster_id'].transform('count')
sampled_negatives_df = negative_df_cleaned.sample(n=NUM_SAMPLES_TO_SELECT, weights=weights, random_state=42).reset_index(drop=True)

print(f"Wybrano {len(sampled_negatives_df)} zróżnicowanych zdań negatywnych.")
print("--- Krok 2 zakończony. ---")
Use code with caution.
Python
Komórka 6: Krok 3 - Montaż finalnego zbioru treningowego
Ta ostatnia komórka bierze cleaned_df i sampled_negatives_df i składa je w ostateczny, gotowy do użycia plik.
Generated python
print("\n--- KROK 3: MONTAŻ FINALNEGO ZBIORU TRENINGOWEGO ---")

# Komponent 1: Oczyszczone zdania pozytywne
cleaned_positives_df = cleaned_df[cleaned_df[RELEVANT_COLUMN] == 1].copy()
# Komponent 2: Zróżnicowane zdania negatywne (już mamy jako 'sampled_negatives_df')

training_data = []
all_unique_labels = cleaned_positives_df[LABEL_COLUMN].unique().tolist()

# 1. Tworzenie przykładów POZYTYWNYCH (is_match = 1)
print("1. Tworzenie przykładów pozytywnych...")
for _, row in cleaned_positives_df.iterrows():
    training_data.append({'text_input': f"{row[LABEL_COLUMN]} [SEP] {row[TEXT_COLUMN]}", 'is_match': 1})

# 2. Tworzenie przykładów "TRUDNYCH" NEGATYWNYCH (is_match = 0)
print("2. Tworzenie 'trudnych' przykładów negatywnych...")
num_hard_negatives = int(len(cleaned_positives_df) * HARD_NEGATIVES_RATIO)
for _ in range(num_hard_negatives):
    row = cleaned_positives_df.sample(n=1).iloc[0]
    wrong_label = random.choice([l for l in all_unique_labels if l != row[LABEL_COLUMN]])
    training_data.append({'text_input': f"{wrong_label} [SEP] {row[TEXT_COLUMN]}", 'is_match': 0})

# 3. Tworzenie przykładów "ŁATWYCH" NEGATYWNYCH (is_match = 0)
print("3. Tworzenie 'łatwych' przykładów negatywnych...")
for _, row in sampled_negatives_df.iterrows():
    random_label = random.choice(all_unique_labels)
    training_data.append({'text_input': f"{random_label} [SEP] {row[TEXT_COLUMN]}", 'is_match': 0})

# 4. Finalizacja i zapis
print("\nFinalizacja i zapis...")
final_training_df = pd.DataFrame(training_data)
final_training_df = final_training_df.sample(frac=1, random_state=42).reset_index(drop=True)

final_training_df.to_csv(OUTPUT_FINAL_TRAINING_DATA_PATH, index=False)

print("\n--- PROCES PRZYGOTOWANIA DANYCH ZAKOŃCZONY POMYŚLNIE! ---")
print(f"Utworzono i zapisano finalny zbiór treningowy zawierający {len(final_training_df)} wierszy.")
print(f"Plik został zapisany w: {OUTPUT_FINAL_TRAINING_DATA_PATH}")

print("\nRozkład etykiet 'is_match' w finalnym zbiorze:")
print(final_training_df['is_match'].value_counts())

print("\nPrzykładowe dane gotowe do treningu:")
display(final_training_df.head(10))
