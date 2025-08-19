# Doskonale. Po wygenerowaniu większej liczby etykiet za pomocą słabej superwizji, naturalnym kolejnym krokiem jest inteligentne wykorzystanie ogromnej puli pozostałych, nieoznaczonych danych. Tutaj właśnie wkraczają Podejścia Pół-nadzorowane (Semi-Supervised) i Aktywne Uczenie (Active Learning).
# Są to dwie różne filozofie wykorzystania nieoznaczonych danych:
# Semi-Supervised Learning: Pozwólmy modelowi samemu się uczyć z nieoznaczonych danych, ufając jego własnym predykcjom. (Automatyczne)
# Active Learning: Niech model wskaże nam, które dane są najbardziej wartościowe do ręcznego oznaczenia, aby zmaksymalizować efektywność pracy człowieka. (Człowiek w pętli)
# Przeanalizujmy oba podejścia wraz z kodem.
# 1. Uczenie Pół-nadzorowane: Pseudo-etykietowanie (Pseudo-Labeling)
# Jest to najprostsza i najbardziej intuicyjna technika uczenia pół-nadzorowanego, nazywana również samouczeniem (self-training).
# Idea:
# Wytrenuj model na swoich najlepszych dostępnych danych (np. 3k ręcznych etykiet + etykiety ze Snorkela).
# Użyj tego modelu, aby przewidzieć etykiety dla wszystkich nieoznaczonych danych.
# Wybierz te predykcje, których model jest najbardziej pewny (np. prawdopodobieństwo dla danej klasy > 95%).
# Dodaj te "pseudo-etykiety" do swojego zbioru treningowego, tak jakby były prawdziwymi etykietami.
# Wytrenuj model od nowa na powiększonym zbiorze.
# Zalety:
# W pełni automatyczny sposób na wykorzystanie całego zbioru danych.
# Może znacząco poprawić wydajność modelu, jeśli jego początkowe predykcje są w miarę dobre.
# Wady:
# Ryzyko propagacji błędu: Jeśli model jest bardzo pewny swojej błędnej predykcji, zacznie uczyć się na własnych błędach, co może pogorszyć jego wydajność. Dlatego kluczowy jest wysoki próg pewności.
# Kod: Przykład Pseudo-etykietowania
# Użyjemy scikit-learn do demonstracji. Załóżmy, że mamy już wstępny zbiór treningowy (oznaczony) i duży zbiór nieoznaczony.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Krok 1: Przygotowanie danych ---

# Załóżmy, że to są Twoje dane po etapie słabej superwizji
# Mamy dane oznaczone i dużą pulę nieoznaczonych
labeled_data = {
    'text': [
        'Ta nowa pomadka ma piękny, matowy kolor.', 'Mój ulubiony tusz do rzęs świetnie wydłuża.',
        'Jaki podkład polecacie do cery mieszanej?', 'Ten krem BB jest lekki, ale dobrze kryje.',
        'Bardzo lubię ten błyszczyk, nie klei się.'
    ],
    'label': ['lipstick', 'mascara', 'foundation', 'foundation', 'lipstick']
}
df_labeled = pd.DataFrame(labeled_data)

unlabeled_data = {
    'text': [
        'Produkt do ust, który utrzymuje się cały dzień.', # Powinno być lipstick
        'Kosmetyk idealnie matuje cerę na wiele godzin.', # Powinno być foundation
        'Idealny do podkreślenia spojrzenia i wydłużenia rzęs.', # Powinno być mascara
        'Kupiłam wczoraj nowy samochód.', # Nierelewantne
        'Czerwona szminka to klasyk w każdej kosmetyczce.', # Powinno być lipstick
    ]
}
df_unlabeled = pd.DataFrame(unlabeled_data)

# Prosty wektoryzator tekstu
vectorizer = TfidfVectorizer()
X_labeled = vectorizer.fit_transform(df_labeled['text'])
y_labeled = df_labeled['label']
X_unlabeled = vectorizer.transform(df_unlabeled['text'])


# --- Krok 2: Wstępne trenowanie modelu ---
print("--- Trenowanie modelu na początkowych danych ---")
model = LogisticRegression()
model.fit(X_labeled, y_labeled)
print(f"Liczba początkowych danych treningowych: {X_labeled.shape[0]}")


# --- Krok 3: Pętla Pseudo-etykietowania ---
print("\n--- Rozpoczęcie procesu pseudo-etykietowania ---")
CONFIDENCE_THRESHOLD = 0.80

# Pobierz prawdopodobieństwa dla nieoznaczonych danych
pred_probas = model.predict_proba(X_unlabeled)

# Znajdź maksymalne prawdopodobieństwo dla każdej predykcji
max_probas = np.max(pred_probas, axis=1)

# Wybierz indeksy tych predykcji, które przekraczają próg
confident_indices = np.where(max_probas >= CONFIDENCE_THRESHOLD)[0]

if len(confident_indices) > 0:
    # Wygeneruj pseudo-etykiety
    pseudo_labels = model.predict(X_unlabeled[confident_indices])
    
    # Dodaj nowe, "pewne" dane do zbioru treningowego
    X_newly_labeled = X_unlabeled[confident_indices]
    y_newly_labeled = pseudo_labels
    
    print(f"Znaleziono {len(pseudo_labels)} pewnych predykcji do dodania.")
    
    # Połącz stare i nowe dane
    X_combined = np.vstack([X_labeled.toarray(), X_newly_labeled.toarray()])
    y_combined = np.concatenate([y_labeled, y_newly_labeled])
    
    # --- Krok 4: Ponowne trenowanie modelu na powiększonym zbiorze ---
    print("\n--- Ponowne trenowanie modelu na połączonych danych ---")
    model.fit(X_combined, y_combined)
    print(f"Nowa liczba danych treningowych: {X_combined.shape[0]}")
    
    # Weryfikacja (dla celów demonstracyjnych)
    print("\nWygenerowane pseudo-etykiety:")
    for i, idx in enumerate(confident_indices):
        print(f"Tekst: '{df_unlabeled['text'].iloc[idx]}', Pseudo-etykieta: '{pseudo_labels[i]}'")
else:
    print("Nie znaleziono żadnych predykcji z wystarczająco wysoką pewnością.")

# 2. Aktywne Uczenie (Active Learning)
# Ta metoda jest idealna, gdy masz możliwość dalszego, ograniczonego etykietowania ręcznego. Zamiast losowo wybierać kolejne dane do oznaczenia, pozwalasz modelowi wybrać te, które przyniosą mu najwięcej "nauki".
# Idea (Cykl Aktywnego Uczenia):
# Wytrenuj model na dostępnych oznaczonych danych.
# Użyj modelu, aby przewidzieć etykiety dla wszystkich nieoznaczonych danych.
# Zamiast szukać pewnych predykcji, znajdź te, których model jest najbardziej niepewny.
# Zaprezentuj te N najbardziej niepewnych przykładów człowiekowi-ekspertowi (Tobie) do ręcznego oznaczenia.
# Dodaj te nowe, wysokiej jakości etykiety do zbioru treningowego.
# Powtórz proces.
# Jak mierzyć niepewność? Najprostsza metoda to "Least Confidence Sampling":
# Niepewność = 1 - (prawdopodobieństwo najbardziej prawdopodobnej klasy)
# Jeśli model przewiduje klasa A: 0.99, klasa B: 0.01, jest bardzo pewny (niepewność = 1 - 0.99 = 0.01).
# Jeśli model przewiduje klasa A: 0.4, klasa B: 0.35, klasa C: 0.25, jest bardzo niepewny (niepewność = 1 - 0.4 = 0.6). Właśnie takie przykłady chcemy etykietować.
# Zalety:
# Maksymalizuje zwrot z inwestycji w ręczne etykietowanie.
# Skupia się na poprawianiu modelu w obszarach, w których jest najsłabszy (na granicach decyzyjnych).
# Wady:
# Wymaga interwencji człowieka w pętli, co spowalnia proces.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Użyjemy tych samych danych co poprzednio
labeled_data = {
    'text': ['Ta nowa pomadka ma piękny, matowy kolor.', 'Mój ulubiony tusz do rzęs świetnie wydłuża.'],
    'label': ['lipstick', 'mascara']
}
df_labeled = pd.DataFrame(labeled_data)

# Pozostałe dane są naszą pulą do etykietowania
pool_data = {
    'text': [
        'Jaki podkład polecacie do cery mieszanej?',
        'Ten krem BB jest lekki, ale dobrze kryje.',
        'Bardzo lubię ten błyszczyk, nie klei się.',
        'Produkt do ust, który utrzymuje się cały dzień.',
        'Kosmetyk idealnie matuje cerę na wiele godzin.',
        'Idealny do podkreślenia spojrzenia i wydłużenia rzęs.'
    ],
    # Te etykiety są "ukryte" dla modelu, ale my je znamy, aby symulować eksperta
    'true_label': ['foundation', 'foundation', 'lipstick', 'lipstick', 'foundation', 'mascara']
}
df_pool = pd.DataFrame(pool_data)

# --- Początkowa konfiguracja ---
vectorizer = TfidfVectorizer()
X_labeled = vectorizer.fit_transform(df_labeled['text'])
y_labeled = df_labeled['label']
X_pool = vectorizer.transform(df_pool['text'])

model = LogisticRegression()

# --- Cykl Aktywnego Uczenia ---
N_QUERIES = 2 # Ile zdań chcemy oznaczyć w jednej iteracji

for i in range(N_QUERIES):
    print(f"\n--- Cykl Aktywnego Uczenia: Iteracja {i+1} ---")
    
    # 1. Trenowanie modelu
    model.fit(X_labeled, y_labeled)
    print(f"Rozmiar zbioru treningowego: {X_labeled.shape[0]}")
    
    # 2. Predykcja na puli nieoznaczonej
    pred_probas = model.predict_proba(X_pool)
    
    # 3. Obliczenie niepewności (Least Confidence)
    max_probas = np.max(pred_probas, axis=1)
    uncertainty_scores = 1 - max_probas
    
    # 4. Wybór najbardziej niepewnego przykładu
    most_uncertain_idx = np.argmax(uncertainty_scores)
    
    text_to_label = df_pool['text'].iloc[most_uncertain_idx]
    print(f"Model jest najbardziej niepewny co do zdania: '{text_to_label}'")
    
    # 5. Symulacja etykietowania przez człowieka
    human_label = df_pool['true_label'].iloc[most_uncertain_idx]
    print(f"Człowiek dostarcza etykietę: '{human_label}'")
    
    # 6. Dodanie nowych danych do zbioru treningowego
    new_X = X_pool[most_uncertain_idx]
    new_y = np.array([human_label])
    
    X_labeled = np.vstack([X_labeled.toarray(), new_X.toarray()])
    y_labeled = np.concatenate([y_labeled, new_y])
    
    # 7. Usunięcie oznaczonego przykładu z puli
    X_pool = np.delete(X_pool.toarray(), most_uncertain_idx, axis=0)
    df_pool = df_pool.drop(df_pool.index[most_uncertain_idx]).reset_index(drop=True)

print(f"\n--- Zakończono. Finalny rozmiar zbioru treningowego: {X_labeled.shape[0]} ---")

# Podsumowanie i Strategia dla Ciebie
# Metoda	Jak działa?	Zalety	Wady	Kiedy używać?
# Pseudo-Labeling	Model uczy się na swoich najpewniejszych predykcjach.	W pełni automatyczne, szybkie skalowanie.	Ryzyko propagacji błędu.	Gdy masz duży zbiór danych po słabej superwizji i chcesz go jeszcze powiększyć bez pracy ręcznej.
# Active Learning	Model prosi człowieka o oznaczenie najbardziej niepewnych przykładów.	Maksymalizuje wartość każdej nowej etykiety, szybko poprawia słabe punkty modelu.	Wymaga człowieka w pętli, wolniejszy proces.	Gdy masz ograniczony budżet/czas na dalsze etykietowanie i chcesz uzyskać jak najlepszy model.
# Rekomendowana strategia hybrydowa:
# Faza 1: Słaba Superwizja (Snorkel): Użyj reguł i kontekstowych LF, aby masowo wygenerować etykiety i powiększyć zbiór z 3k do np. 20-30k. To jest Twoja podstawa.
# Faza 2: Uczenie Pół-nadzorowane (Pseudo-Labeling): Wytrenuj solidny model (np. BERT) na danych z Fazy 1. Następnie użyj go do wygenerowania pseudo-etykiet dla pozostałych 60k danych, ale z bardzo wysokim progiem pewności (np. 0.98), aby zminimalizować ryzyko.
# Faza 3: Aktywne Uczenie (Active Learning): Na koniec, przeprowadź kilka cykli aktywnego uczenia. Pozwól swojemu potężnemu modelowi wskazać 100-200 zdań, co do których jest najbardziej zdezorientowany. Ręczne oznaczenie tych kilku krytycznych przykładów może "załatać" najważniejsze dziury w jego wiedzy.
