import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# ======================================================================================
# GŁÓWNY I OSTATECZNY BLOK KODU DO WERYFIKACJI I KOREKTY 3000 ETYKIET
# ======================================================================================

# Krok 1: Przygotowanie danych i wczytanie modelu
print("🔄 Przygotowywanie danych i wczytywanie modelu...")
# Upewnij się, że model jest wczytany.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"Model wczytany. Weryfikacja będzie wykonana na: {device.upper()}")

# Wybierz tylko te wiersze z Twojego DataFrame, które mają etykietę
df_labeled = df.dropna(subset=['label']).copy()

# Wyodrębnij unikalne etykiety, które będą pulą kandydatów
candidate_labels = df_labeled['label'].unique().tolist()
print(f"✅ Znaleziono {len(df_labeled)} oznaczonych wierszy i {len(candidate_labels)} unikalnych etykiet do weryfikacji.")

# Krok 2: Tworzenie wektorów (embeddings) dla tekstów i etykiet
print("\n🔄 Przetwarzanie tekstów i etykiet na wektory numeryczne...")
# To jest najdłuższy krok, ale na GPU pójdzie szybko.
text_embeddings = model.encode(df_labeled['text'].astype(str).tolist(), batch_size=256, show_progress_bar=True)
label_embeddings = model.encode(candidate_labels, batch_size=256, show_progress_bar=True)
print("✅ Teksty i etykiety przetworzone.")

# Krok 3: Znalezienie najlepszej etykiety dla każdego tekstu
print("\n🔄 Sprawdzanie każdego wiersza i szukanie najlepszego dopasowania...")
# Oblicz podobieństwo każdego tekstu do WSZYSTKICH możliwych etykiet
similarity_scores = util.cos_sim(text_embeddings, label_embeddings)

# Przygotuj listy na nowe kolumny
model_suggestions = []
confidence_scores = []

# Przejdź przez każdy wiersz i znajdź najlepszy wynik
for i in range(len(df_labeled)):
    best_match_index = torch.argmax(similarity_scores[i]).item()
    best_label = candidate_labels[best_match_index]
    best_score = similarity_scores[i][best_match_index].item() * 100
    
    model_suggestions.append(best_label)
    confidence_scores.append(f"{best_score:.2f}%")

# Krok 4: Dodanie nowych kolumn do DataFrame z etykietami
df_labeled['sugestia_modelu'] = model_suggestions
df_labeled['pewnosc_sugestii_%'] = confidence_scores

# Stwórz kolumnę 'nowy_label', która pokazuje tylko zmiany
# np.where(warunek, wartość_jeśli_prawda, wartość_jeśli_fałsz)
df_labeled['nowy_label'] = np.where(
    df_labeled['label'] != df_labeled['sugestia_modelu'], # warunek: oryginalna etykieta jest inna niż sugestia
    df_labeled['sugestia_modelu'],                        # jeśli tak, wstaw sugestię
    None                                                 # jeśli nie, zostaw puste (None)
)
print("✅ Analiza zakończona. Dodano nowe kolumny.")

# Krok 5: Aktualizacja oryginalnego DataFrame 'df'
# Użyjemy .update(), aby przenieść nowe kolumny do głównego DataFrame, dopasowując po indeksie
df.update(df_labeled)

# Krok 6: Wyświetlenie raportu ze zmianami
print("\n\n--- WYNIKI WERYFIKACJI ---")
# Wybierz tylko te wiersze, gdzie model znalazł błąd
df_corrections = df[df['nowy_label'].notna()].copy()

if df_corrections.empty:
    print("✅ Model nie znalazł żadnych etykiet do poprawy. Wszystkie 3000 są zgodne z jego sugestiami.")
else:
    print(f"✅ Model znalazł {len(df_corrections)} etykiet do poprawy.")
    print("Poniżej znajduje się lista tych wierszy:")
    
    # Ustawienie opcji wyświetlania, żeby tekst nie był ucinany
    pd.set_option('display.max_colwidth', None)
    
    # Wyświetl najważniejsze kolumny dla wierszy, które zostały zmienione
    print(df_corrections[['text', 'label', 'sugestia_modelu', 'pewnosc_sugestii_%']])
