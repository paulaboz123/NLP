import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import defaultdict

# ===================================================================
# GŁÓWNY BLOK OBLICZENIOWY - WYKONYWANY NA ZDALNYM GPU
# ===================================================================

# Krok 1: Weryfikacja GPU i wczytanie modelu
# To jest natychmiastowy dowód, że kod działa na właściwej maszynie.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("⚠️ OSTRZEŻENIE: Nie wykryto GPU. Obliczenia będą wolne. Upewnij się, że kernel notatnika jest połączony z maszyną GPU.")
else:
    print(f"✅ Potwierdzono! Kod jest wykonywany na urządzeniu: {device.upper()}")

# Wczytanie modelu SBERT bezpośrednio na GPU
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("✅ Model SBERT wczytany na GPU.")

# Krok 2: Tworzenie prototypów z oznaczonych danych
# Ten proces jest bardzo szybki.
print("\n🔄 Tworzenie semantycznych prototypów z oznaczonych danych...")
labeled_df = df.dropna(subset=[LABEL_COLUMN]).copy()
labeled_texts = labeled_df[TEXT_COLUMN].astype(str).tolist()

# Obliczenia na GPU
labeled_embeddings = model.encode(labeled_texts, batch_size=512, show_progress_bar=True)

# Agregacja do prototypów
label_embeddings_dict = defaultdict(list)
for i in range(len(labeled_df)):
    label = labeled_df.iloc[i][LABEL_COLUMN]
    embedding = labeled_embeddings[i]
    label_embeddings_dict[label].append(embedding)

prototype_embeddings_list = []
prototype_labels = []
for label, embeddings in label_embeddings_dict.items():
    prototype_embeddings_list.append(np.mean(embeddings, axis=0))
    prototype_labels.append(label)

prototype_embeddings = torch.tensor(np.array(prototype_embeddings_list), device=device)
print(f"✅ Stworzono {len(prototype_labels)} prototypów.")


# Krok 3: Kodowanie wszystkich nieoznaczonych zdań
# To jest główna, ciężka operacja, która teraz będzie bardzo szybka na GPU.
unlabeled_df = df[df[LABEL_COLUMN].isnull()].copy()
unlabeled_texts = unlabeled_df[TEXT_COLUMN].astype(str).tolist()
print(f"\n🔄 Kodowanie {len(unlabeled_texts)} nieoznaczonych zdań na GPU...")

# Obliczenia na GPU
unlabeled_embeddings = model.encode(unlabeled_texts, batch_size=512, show_progress_bar=True)
print("✅ Kodowanie zakończone.")


# Krok 4: Wyszukiwanie semantyczne i zastosowanie zmian
# Ta operacja jest również ekstremalnie szybka na GPU.
print("\n🔄 Wyszukiwanie najbliższych etykiet i stosowanie zmian...")
hits = util.semantic_search(torch.tensor(unlabeled_embeddings).to(device), prototype_embeddings, top_k=1)

update_indices = []
new_labels = []
unlabeled_df_indices = unlabeled_df.index

for i, hit_list in enumerate(hits):
    if hit_list and hit_list[0]['score'] >= CONFIDENCE_THRESHOLD:
        original_df_index = unlabeled_df_indices[i]
        update_indices.append(original_df_index)
        new_labels.append(prototype_labels[hit_list[0]['corpus_id']])

if update_indices:
    # Policz, ile flag 'RELEVANT' zostanie zmienionych, zanim je zmienisz
    flipped_relevance_count = (df.loc[update_indices, RELEVANT_COLUMN] == 0).sum()
    
    # Zastosuj zmiany w DataFrame w jednej operacji
    df.loc[update_indices, LABEL_COLUMN] = new_labels
    df.loc[update_indices, RELEVANT_COLUMN] = 1
    
    print(f"\n✅ Zastosowano {len(update_indices)} nowych etykiet.")
    print(f"✅ Zmieniono flagę '{RELEVANT_COLUMN}' z 0 na 1 dla {flipped_relevance_count} wierszy.")
else:
    print("\n⚠️ Żadne zdanie nie osiągnęło progu pewności. Nie wprowadzono żadnych zmian.")

# Krok 5: Wyświetlenie wyników
print("\n\n--- WYNIKI KOŃCOWE ---")
print("Nowy rozkład wartości w kolumnie 'RELEVANT':")
print(df[RELEVANT_COLUMN].value_counts())
print(f"\nLiczba wierszy, które nadal nie mają etykiety: {df[LABEL_COLUMN].isnull().sum()}")
print("\n✅✅✅ Przetwarzanie zakończone.")
