# Wczytanie modelu SBERT
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# --- Tworzenie prototypów (błyskawiczne) ---
print("\nTworzenie semantycznych prototypów z oznaczonych danych...")
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
print(f"✅ Stworzono {len(prototype_labels)} prototypów.")

# --- Kodowanie nieoznaczonych zdań (szybkie na GPU) ---
unlabeled_df = df[df["DEMAND_ID"].isnull()].copy()
unlabeled_texts = unlabeled_df["TEXT"].astype(str).tolist()
print(f"\nKodowanie {len(unlabeled_texts)} nieoznaczonych zdań na GPU...")
unlabeled_embeddings = model.encode(unlabeled_texts, batch_size=512, show_progress_bar=True)

# --- Wyszukiwanie semantyczne (błyskawiczne na GPU) ---
print("\nWyszukiwanie najbliższych etykiet...")
hits = util.semantic_search(torch.tensor(unlabeled_embeddings).to(device), prototype_embeddings, top_k=1)

# --- Zastosowanie zmian ---
update_indices = []
new_labels = []
unlabeled_df_indices = unlabeled_df.index

for i, hit_list in enumerate(hits):
    if hit_list and hit_list[0]['score'] >= CONFIDENCE_THRESHOLD:
        original_df_index = unlabeled_df_indices[i]
        update_indices.append(original_df_index)
        new_labels.append(prototype_labels[hit_list[0]['corpus_id']])

if update_indices:
    flipped_relevance_count = (df.loc[update_indices, "RELEVANT"] == 0).sum()
    df.loc[update_indices, "DEMAND_ID"] = new_labels
    df.loc[update_indices, "RELEVANT"] = 1
    print(f"\n✅ Zastosowano {len(update_indices)} nowych etykiet.")
    print(f"✅ Zmieniono flagę 'RELEVANT' z 0 na 1 dla {flipped_relevance_count} wierszy.")
else:
    print("\n⚠️ Żadne zdanie nie osiągnęło progu pewności. Nie wprowadzono żadnych zmian.")

# --- STAN DANYCH (PO PRZETWORZENIU) ---
print("\n--- STAN DANYCH (PO PRZETWORZENIU) ---")
print("Liczba wierszy w kolumnie 'RELEVANT':")
print(df["RELEVANT"].value_counts())
print(f"\nLiczba wierszy z pustą etykietą 'DEMAND_ID': {df['DEMAND_ID'].isnull().sum()}")

# Zapisanie przetworzonego pliku
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n\n✅✅✅ WSZYSTKO GOTOWE! Wynikowy plik został zapisany jako '{OUTPUT_FILE}'.")
print("Możesz go teraz pobrać z panelu 'Notebooks' w Azure ML Studio, klikając na niego i wybierając 'Download'.")
