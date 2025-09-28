# Uruchom tę komórkę tylko raz
!pip install -q sentence-transformers==3.0.1 torch==2.3.1 pandas==2.2.2

import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Wczytujemy model. Jeśli masz GPU, automatycznie go użyje.
model = SentenceTransformer('all-MiniLM-L6-v2')

print("✅ Model SBERT gotowy do pracy.")

# --- KROK 1: Przygotowanie danych bezpośrednio z Twojego DataFrame ---

# Pobierz unikalne, istniejące etykiety z kolumny 'label'. Ignoruje puste wartości (NaN).
istniejace_etykiety = df['label'].dropna().unique().tolist()

# Wybierz wiersze, które potrzebują etykiety:
# - 'label' jest pusty (isnull)
# - 'relevant' jest równe 1
df_do_oznaczenia = df[df['label'].isnull() & (df['relevant'] == 1)].copy()

# Pobierz teksty z tych wierszy
zdania_do_oznaczenia = df_do_oznaczenia['text'].tolist()

print(f"Znaleziono {len(istniejace_etykiety)} unikalnych etykiet w Twoich danych.")
print(f"Znaleziono {len(zdania_do_oznaczenia)} istotnych zdań, które potrzebują etykiety.")


# --- KROK 2: Uruchomienie modelu SBERT (jeśli są jakieś zdania do oznaczenia) ---

if not zdania_do_oznaczenia:
    print("\nNie znaleziono żadnych istotnych zdań do etykietowania. Praca zakończona.")
else:
    # Konwersja na wektory (embeddings)
    etykiety_embeddings = model.encode(istniejace_etykiety, convert_to_tensor=True)
    zdania_embeddings = model.encode(zdania_do_oznaczenia, convert_to_tensor=True)

    # Obliczenie podobieństwa i znalezienie najlepszego dopasowania dla każdego zdania
    wyniki_podobienstwa = util.cos_sim(zdania_embeddings, etykiety_embeddings)

    przewidziane_etykiety = []
    poziom_pewnosci = []

    for i in range(len(zdania_do_oznaczenia)):
        # Znajdź indeks etykiety z najwyższym wynikiem
        najlepszy_index = torch.argmax(wyniki_podobienstwa[i])
        # Pobierz nazwę tej etykiety
        najlepsza_etykieta = istniejace_etykiety[najlepszy_index]
        # Pobierz wynik (pewność) i zamień na procent
        pewnosc = wyniki_podobienstwa[i][najlepszy_index] * 100
        
        przewidziane_etykiety.append(najlepsza_etykieta)
        poziom_pewnosci.append(f"{pewnosc:.2f}%")

    # --- KROK 3: Aktualizacja oryginalnego DataFrame 'df' ---
    
    # Dodaj nowe kolumny do tymczasowego DataFrame
    df_do_oznaczenia['PRZEWIDZIANA_ETYKIETA'] = przewidziane_etykiety
    df_do_oznaczenia['PEWNOSC_%'] = poziom_pewnosci
    
    # Zaktualizuj główny DataFrame 'df' na podstawie indeksu
    # To zapewni, że wyniki trafią do właściwych wierszy
    df.update(df_do_oznaczenia)
    
    # Opcjonalnie: od razu wypełnij pustą kolumnę 'label' nowymi przewidywaniami
    df['label'].fillna(df['PRZEWIDZIANA_ETYKIETA'], inplace=True)

    print("\n✅ Etykietowanie zakończone.")


# --- WYNIK KOŃCOWY ---
print("\n--- Podgląd zaktualizowanego DataFrame (pierwsze 10 wierszy) ---")
# Wyświetlamy tylko wiersze, które zostały właśnie zaktualizowane, żeby było widać efekt
print(df[df.index.isin(df_do_oznaczenia.index)].head(10))

print("\n--- Podgląd całego DataFrame (pierwsze 10 wierszy) ---")
print(df.head(10))
