# =================================================================================
# SCRIPT: Szybkie Etykietowanie i Korekta Danych za pomocą Wyszukiwania Semantycznego
# Wersja zoptymalizowana pod kątem szybkości na CPU (cel: < 3 godziny)
# =================================================================================

# Step 0: Instalacja niezbędnych pakietów.
# Uruchom tę komórkę tylko raz.
# !pip install -q pandas==2.2.2 torch==2.3.1 sentence-transformers==3.0.1 scikit-learn==1.5.0

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import sys
from collections import defaultdict

print("--- Start: Szybki proces etykietowania i korekty RELEVANT (zoptymalizowany dla CPU) ---")

# --- KONFIGURACJA (WYMAGANA AKCJA) ---
# 1. Ustaw ścieżkę do swojego pliku z danymi.
FILE_PATH = "your_dataset.csv"  # <--- ZMIEŃ NA ŚCIEŻKĘ DO TWOJEGO PLIKU

# 2. Potwierdź nazwy kolumn.
TEXT_COLUMN = "TEXT"
LABEL_COLUMN = "DEMAND_ID"
RELEVANT_COLUMN = "RELEVANT"

# 3. Ustaw próg pewności (podobieństwa semantycznego) do przypisania etykiety.
# Dla tej metody próg może być niższy. Dobry punkt startowy to 0.5.
# Jeśli dostaniesz za mało etykiet, zmniejsz go. Jeśli za dużo błędnych, zwiększ.
CONFIDENCE_THRESHOLD = 0.5

# 4. Ustaw nazwę dla pliku wynikowego.
OUTPUT_FILE_PATH = "your_dataset_FAST_LABELED.csv"
# --- KONIEC KONFIGURACJI ---


# Krok 1: Wczytanie zbioru danych
try:
    df = pd.read_csv(FILE_PATH)
    print(f"\n✅ Pomyślnie wczytano zbiór danych z {len(df)} wierszami z '{FILE_PATH}'.")
    df[RELEVANT_COLUMN] = df[RELEVANT_COLUMN].astype(int)
except FileNotFoundError:
    print(f"❌ BŁĄD: Plik '{FILE_PATH}' nie został znaleziony. Sprawdź ścieżkę i spróbuj ponownie.")
    sys.exit()

# --- WERYFIKACJA (PRZED) ---
print("\n--- STAN DANYCH (PRZED SKRYPTEM) ---")
print(f"Liczba wierszy w kolumnie '{RELEVANT_COLUMN}':")
print(df[RELEVANT_COLUMN].value_counts())
print(f"\nLiczba wierszy z pustą etykietą '{LABEL_COLUMN}': {df[LABEL_COLUMN].isnull().sum()}")
# ---

# Krok 2: Wczytanie modelu SBERT
device = 'cpu' # Jawnie ustawiamy CPU
model_name = 'all-MiniLM-L6-v2' # Szybki i dobry model
print(f"\n🔄 Wczytywanie modelu SBERT '{model_name}' na urządzenie '{device}'...")
model = SentenceTransformer(model_name, device=device)
print("✅ Model wczytany pomyślnie.")

# Krok 3: Stworzenie "Prototypów" dla każdej etykiety (zamiast długiego treningu)
print("\n🔄 Tworzenie semantycznych prototypów z Twoich 3000 oznaczonych przykładów...")
labeled_df = df.dropna(subset=[LABEL_COLUMN]).copy()

# Słownik do przechowywania wektorów dla każdej etykiety
label_embeddings_dict = defaultdict(list)

# Zakoduj wszystkie teksty z oznaczonych danych
# Używamy batch_size, aby efektywnie zarządzać pamięcią
labeled_texts = labeled_df[TEXT_COLUMN].astype(str).tolist()
labeled_embeddings = model.encode(labeled_texts, batch_size=128, show_progress_bar=True)

# Grupuj wektory według etykiety
for i, row in labeled_df.iterrows():
    label = row[LABEL_COLUMN]
    embedding = labeled_embeddings[i]
    label_embeddings_dict[label].append(embedding)

# Oblicz średni wektor (prototyp) dla każdej etykiety
prototype_embeddings = []
prototype_labels = []
for label, embeddings in label_embeddings_dict.items():
    prototype_embeddings.append(np.mean(embeddings, axis=0))
    prototype_labels.append(label)

prototype_embeddings = torch.tensor(np.array(prototype_embeddings), device=device)
print(f"✅ Stworzono {len(prototype_labels)} prototypów.")

# Krok 4: Zakodowanie wszystkich nieoznaczonych zdań (krok zoptymalizowany)
print(f"\n🔄 Kodowanie {df[LABEL_COLUMN].isnull().sum()} nieoznaczonych zdań. To jest główny, najdłuższy krok...")
print("💡 Używamy przetwarzania równoległego, aby maksymalnie przyspieszyć ten proces.")

unlabeled_texts = df[df[LABEL_COLUMN].isnull()][TEXT_COLUMN].astype(str).tolist()

# === KLUCZOWA OPTYMALIZACJA PRĘDKOŚCI ===
# Uruchamiamy pule procesów, aby użyć wielu rdzeni CPU
pool = model.start_multi_process_pool()
# Kodujemy zdania, praca jest automatycznie rozdzielana na wiele procesów
unlabeled_embeddings = model.encode_multi_process(unlabeled_texts, pool, batch_size=128)
model.stop_multi_process_pool(pool)
# ==========================================

unlabeled_embeddings = torch.tensor(unlabeled_embeddings, device=device)
print("✅ Kodowanie nieoznaczonych zdań zakończone.")

# Krok 5: Wyszukiwanie semantyczne - znalezienie najbliższego prototypu dla każdego zdania
print("\n🔄 Wyszukiwanie najbliższych etykiet dla każdego zdania...")
# util.semantic_search jest ekstremalnie szybką operacją
hits = util.semantic_search(unlabeled_embeddings, prototype_embeddings, top_k=1)
print("✅ Wyszukiwanie zakończone.")

# Krok 6: Zastosowanie zmian na podstawie progu pewności
print(f"\n🔄 Stosowanie zmian dla wyników z pewnością >= {CONFIDENCE_THRESHOLD}...")
update_indices = []
new_labels = []
relevance_flipped_count = 0

unlabeled_df_indices = df[df[LABEL_COLUMN].isnull()].index

for i, hit_list in enumerate(hits):
    if not hit_list:
        continue
    
    best_hit = hit_list[0]
    confidence_score = best_hit['score']
    
    if confidence_score >= CONFIDENCE_THRESHOLD:
        original_df_index = unlabeled_df_indices[i]
        
        # Zapisz indeks i nową etykietę
        update_indices.append(original_df_index)
        new_labels.append(prototype_labels[best_hit['corpus_id']])
        
        # Sprawdź, czy RELEVANT zostanie zmieniony
        if df.loc[original_df_index, RELEVANT_COLUMN] == 0:
            relevance_flipped_count += 1

# Zastosuj zmiany w DataFrame w jednej, szybkiej operacji
if update_indices:
    df.loc[update_indices, LABEL_COLUMN] = new_labels
    df.loc[update_indices, RELEVANT_COLUMN] = 1
    print(f"✅ Zastosowano {len(update_indices)} nowych etykiet.")
    print(f"✅ Zmieniono flagę '{RELEVANT_COLUMN}' z 0 na 1 dla {relevance_flipped_count} wierszy.")
else:
    print("\n⚠️ Żadne zdanie nie osiągnęło progu pewności. Nie wprowadzono żadnych zmian.")
    print("💡 Spróbuj obniżyć CONFIDENCE_THRESHOLD na górze skryptu, jeśli oczekujesz więcej dopasowań.")

# --- WERYFIKACJA (PO) ---
print("\n--- STAN DANYCH (PO SKRYPCIE) ---")
print(f"Liczba wierszy w kolumnie '{RELEVANT_COLUMN}' (powinna być zaktualizowana):")
print(df[RELEVANT_COLUMN].value_counts())
print(f"\nLiczba wierszy z pustą etykietą '{LABEL_COLUMN}': {df[LABEL_COLUMN].isnull().sum()}")
# ---

# Krok 7: Zapisanie wyniku końcowego
df.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"\n✅ Wszystko gotowe! Zaktualizowany zbiór danych został zapisany w: '{OUTPUT_FILE_PATH}'")
