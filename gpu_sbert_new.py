import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import re

# ======================================================================================
# GŁÓWNY BLOK KODU DO ETYKIETOWANIA Z AUTOMATYCZNYM CZYSZCZENIEM DANYCH
# ======================================================================================

# --- Funkcja do czyszczenia tekstu ---
def czysc_tekst(tekst):
    # Upewnij się, że przetwarzamy string
    if not isinstance(tekst, str):
        return ""
    # 1. Usuń znaki specjalne (zostaw litery, cyfry i spacje)
    tekst = re.sub(r'[^a-zA-Z0-9\s]', '', tekst)
    # 2. Zamień wielokrotne spacje na pojedynczą
    tekst = re.sub(r'\s+', ' ', tekst).strip()
    # 3. Konwertuj na małe litery
    tekst = tekst.lower()
    return tekst
# ------------------------------------

# Krok 1: Przygotowanie danych i wczytanie modelu
print("🔄 Przygotowywanie danych i wczytywanie modelu...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"Model wczytany. Etykietowanie będzie wykonane na: {device.upper()}")

# Wyodrębnij unikalne etykiety z 3k oznaczonych wierszy
candidate_labels = df.dropna(subset=['label'])['label'].unique().tolist()

# Wybierz tylko wiersze, które nie mają etykiety
df_to_label = df[df['label'].isnull()].copy()

# Krok 2: Czyszczenie i filtrowanie danych
print(f"\n🔄 Czyszczenie {len(df_to_label)} wierszy przed analizą...")
# Stwórz nową, tymczasową kolumnę z wyczyszczonym tekstem
df_to_label['text_cleaned'] = df_to_label['text'].apply(czysc_tekst)

# Oblicz liczbę słów w wyczyszczonym tekście
df_to_label['word_count'] = df_to_label['text_cleaned'].apply(lambda x: len(x.split()))

# Filtruj DataFrame, zostawiając tylko wiersze z co najmniej 3 słowami
df_to_label_filtered = df_to_label[df_to_label['word_count'] >= 3].copy()
rows_ignored_count = len(df_to_label) - len(df_to_label_filtered)

print(f"✅ Czyszczenie zakończone. {rows_ignored_count} wierszy zostało zignorowanych (mniej niż 3 słowa).")

# Pobierz ostateczną listę tekstów do analizy
texts_to_label = df_to_label_filtered['text_cleaned'].tolist()

if not texts_to_label:
    print("\n✅ Nie znaleziono żadnych wierszy (z co najmniej 3 słowami) do etykietowania. Praca zakończona.")
else:
    print(f"✅ Znaleziono {len(candidate_labels)} unikalnych etykiet do użycia.")
    print(f"✅ {len(texts_to_label)} wierszy zostanie automatycznie oznaczonych.")

    # Krok 3: Tworzenie wektorów (embeddings)
    print("\n🔄 Przetwarzanie tekstów i etykiet na wektory numeryczne...")
    label_embeddings = model.encode(candidate_labels, batch_size=256, show_progress_bar=True)
    # Używamy wyczyszczonych tekstów do kodowania!
    text_embeddings = model.encode(texts_to_label, batch_size=512, show_progress_bar=True)
    print("✅ Teksty i etykiety przetworzone.")

    # Krok 4: Znalezienie najlepszej etykiety dla każdego tekstu
    print("\n🔄 Wyszukiwanie najlepszego dopasowania dla każdego nieoznaczonego wiersza...")
    similarity_scores = util.cos_sim(text_embeddings, label_embeddings)

    model_suggestions = []
    confidence_scores = []

    for i in range(len(texts_to_label)):
        best_match_index = torch.argmax(similarity_scores[i]).item()
        best_label = candidate_labels[best_match_index]
        best_score = similarity_scores[i][best_match_index].item() * 100
        
        model_suggestions.append(best_label)
        confidence_scores.append(f"{best_score:.2f}%")
    
    # Krok 5: Dodanie nowych kolumn do przefiltrowanego DataFrame
    df_to_label_filtered['sugerowany_label'] = model_suggestions
    df_to_label_filtered['pewnosc_sugestii_%'] = confidence_scores
    
    # Krok 6: Aktualizacja oryginalnego DataFrame 'df'
    print("✅ Analiza zakończona. Aktualizowanie głównego DataFrame...")
    df.update(df_to_label_filtered)
    
    # Krok 7: Automatyczne wypełnienie pustej kolumny 'label' nowymi sugestiami
    df['label'].fillna(df['sugerowany_label'], inplace=True)

    # Krok 8: Wyświetlenie raportu
    print("\n\n--- WYNIKI ETYKIETOWANIA ---")
    print("Poniżej znajduje się podgląd wierszy, które zostały właśnie oznaczone:")
    
    pd.set_option('display.max_colwidth', None)
    
    # Wyświetl najważniejsze kolumny dla wierszy, które zostały właśnie zmienione
    print(df[df.index.isin(df_to_label_filtered.index)].head(10))

    print(f"\n✅✅✅ Proces zakończony. {len(df_to_label_filtered)} wierszy zostało oznaczonych.")
