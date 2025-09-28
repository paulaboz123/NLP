import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# ======================================================================================
# GŁÓWNY I OSTATECZNY BLOK KODU DO ETYKIETOWANIA I KORYGOWANIA DANYCH
# ======================================================================================

# --- Konfiguracja (możesz dostosować) ---
# Próg pewności. Jeśli dopasowanie jest poniżej tej wartości, zdanie zostanie zignorowane.
# Dobry punkt startowy to 0.6. Zwiększ, jeśli dostajesz złe etykiety; zmniejsz, jeśli za mało.
CONFIDENCE_THRESHOLD = 0.6
# ------------------------------------

# Krok 1: Wczytanie modelu i weryfikacja GPU (jeśli dostępne)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"Model SBERT wczytany. Obliczenia będą wykonywane na: {device.upper()}")

# Krok 2: Tworzenie prototypów z Twoich 3k oznaczonych danych
print("\n🔄 Tworzenie profili znaczeniowych dla istniejących etykiet...")
labeled_df = df.dropna(subset=['label']).copy()
labeled_texts = labeled_df['text'].astype(str).tolist()
labeled_embeddings = model.encode(labeled_texts, batch_size=256, show_progress_bar=True)

label_embeddings_dict = defaultdict(list)
for i in range(len(labeled_df)):
    label = labeled_df.iloc[i]['label']
    embedding = labeled_embeddings[i]
    label_embeddings_dict[label].append(embedding)

prototype_embeddings_list = []
prototype_labels = []
for label, embeddings in label_embeddings_dict.items():
    prototype_embeddings_list.append(np.mean(embeddings, axis=0))
    prototype_labels.append(label)

prototype_embeddings = torch.tensor(np.array(prototype_embeddings_list), device=device)
print(f"✅ Stworzono {len(prototype_labels)} profili etykiet.")

# Krok 3: Przetwarzanie wszystkich nieoznaczonych wierszy
# To jest kluczowy krok: bierzemy WSZYSTKIE wiersze bez etykiety, niezależnie od flagi 'relevant'
rows_to_process = df[df['label'].isnull()].copy()
texts_to_process = rows_to_process['text'].astype(str).tolist()

if not texts_to_process:
    print("\nBrak wierszy do przetworzenia.")
else:
    print(f"\n🔄 Analizowanie {len(texts_to_process)} nieoznaczonych zdań...")
    unlabeled_embeddings = model.encode(texts_to_process, batch_size=512, show_progress_bar=True)
    
    # Znajdowanie najlepszych dopasowań
    hits = util.semantic_search(torch.tensor(unlabeled_embeddings).to(device), prototype_embeddings, top_k=1)
    
    update_indices = []
    new_labels = []

    for i, hit_list in enumerate(hits):
        # Sprawdzamy, czy dopasowanie jest wystarczająco dobre
        if hit_list and hit_list[0]['score'] >= CONFIDENCE_THRESHOLD:
            # Pobieramy oryginalny indeks wiersza z głównego DataFrame
            original_df_index = rows_to_process.index[i]
            
            update_indices.append(original_df_index)
            new_labels.append(prototype_labels[hit_list[0]['corpus_id']])

    # Krok 4: Zastosowanie zmian w głównym DataFrame 'df'
    if update_indices:
        # Policz, ile flag 'relevant' zostanie zmienionych
        flipped_relevance_count = (df.loc[update_indices, 'relevant'] == 0).sum()
        
        print(f"\n✅ Znaleziono {len(update_indices)} pasujących zdań, które zostaną zaktualizowane.")
        print(f"   - Zostanie zmieniona flaga 'relevant' z 0 na 1 dla {flipped_relevance_count} wierszy.")
        
        # Zastosuj zmiany w jednej operacji
        df.loc[update_indices, 'label'] = new_labels
        df.loc[update_indices, 'relevant'] = 1
    else:
        print("\n⚠️ Żadne zdanie nie osiągnęło wymaganego progu pewności. Nie wprowadzono żadnych zmian.")


# --- WYNIK KOŃCOWY ---
print("\n\n--- WYNIKI PRZETWARZANIA ---")
print("Nowy rozkład wartości w kolumnie 'relevant' (powinien się zmienić):")
print(df['relevant'].value_counts())
print(f"\nLiczba wierszy, które nadal nie mają etykiety: {df['label'].isnull().sum()}")

# Wyświetl przykładowe wiersze, które zostały właśnie zmienione
if update_indices:
    print("\nPrzykładowe wiersze, które zostały zaktualizowane:")
    print(df.loc[update_indices].head())
