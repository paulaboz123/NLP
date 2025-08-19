# Oczywiście. Skupmy się teraz szczegółowo na Uczeniu Pół-nadzorowanym (Semi-Supervised Learning), a konkretnie na jego najbardziej intuicyjnej i powszechnie stosowanej formie: Samouczeniu (Self-Training) poprzez Pseudo-etykietowanie (Pseudo-Labeling).
# Ta metoda jest niezwykle potężnym narzędziem w Twoim arsenale, ponieważ pozwala w sposób w pełni zautomatyzowany wykorzystać wartość ukrytą w Twoich 87 000 nieoznaczonych zdań.
# Główna Idea: Model jako Ostrożny Student
# Wyobraź sobie model jak studenta, który przygotowuje się do bardzo trudnego egzaminu z 200 tematów.
# Materiały z Odpowiedziami: Na początku dostaje 3 000 zadań z kluczem odpowiedzi (Twoje ręcznie oznaczone dane). Uczy się na nich tak dobrze, jak potrafi.
# Zeszyt Ćwiczeń bez Odpowiedzi: Następnie dostaje ogromny zeszyt z 87 000 dodatkowych zadań (Twoje nieoznaczone dane). Rozwiązuje je wszystkie, ale nie wie, czy zrobił to dobrze.
# Weryfikacja Własnej Wiedzy: Student jest jednak mądry i ostrożny. Przegląda swoje rozwiązania i wybiera tylko te, co do których jest absolutnie pewien (np. na 99% jest przekonany, że odpowiedź jest prawidłowa).
# Tworzenie "Pewniaków": Te kilka zadań, których jest super-pewien, traktuje tak, jakby pochodziły z oryginalnego klucza odpowiedzi. Dopisuje je do swoich materiałów do nauki.
# Ponowna Nauka: Teraz uczy się ponownie, ale już na większym zbiorze materiałów – oryginalnym kluczu plus swoich własnych, zweryfikowanych "pewniakach".
# Te "pewniaki" to właśnie pseudo-etykiety.
# Proces Krok po Kroku
# Proces jest iteracyjny i opiera się na stopniowym budowaniu zaufania do własnych predykcji modelu.
# Trenowanie Modelu Bazowego: Wytrenuj swój najlepszy możliwy model (np. BERT dostrojony do klasyfikacji wieloetykietowej) na najlepszych dostępnych danych (Twoje 3k ręcznych etykiet + wysokiej jakości etykiety ze słabej superwizji, np. Snorkela).
# Predykcja na Nieoznaczonych Danych: Użyj tego modelu, aby przewidzieć prawdopodobieństwa dla każdej z 200 klas dla wszystkich zdań w Twojej nieoznaczonej puli.
# Selekcja na Podstawie Pewności: Ustal próg pewności (confidence threshold), np. 0.95. To jest najważniejszy parametr. Przejrzyj wszystkie predykcje i wybierz te, w których maksymalne prawdopodobieństwo dla jakiejkolwiek klasy przekroczyło ten próg.
# Generowanie Pseudo-Etykiet: Dla każdego zdania, które przeszło selekcję, przypisz mu etykietę o najwyższym prawdopodobieństwie. To są Twoje pseudo-etykiety.
# Powiększenie Zbioru Treningowego: Dodaj zdania z pseudo-etykietami do swojego oryginalnego, "złotego" zbioru treningowego.
# Ponowne Trenowanie: Wytrenuj model od nowa na tym nowym, powiększonym zbiorze danych. Model ten powinien być już nieco lepszy, ponieważ uczył się na większej ilości (choć nieco bardziej zaszumionych) danych.
# (Opcjonalnie) Iteracja: Możesz powtórzyć kroki 2-6, używając nowo wytrenowanego, lepszego modelu do etykietowania pozostałych danych.
# Kod: Iteracyjny Proces Samouczenia
# Poniższy kod demonstruje, jak można zaimplementować pętlę samouczenia w scikit-learn.


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Krok 1: Przygotowanie Danych ---
# Załóżmy, że df_labeled to Twoje "złote" dane, a df_unlabeled to reszta
all_data = {
    'text': [
        'Ta nowa pomadka ma piękny, matowy kolor.', 'Mój ulubiony tusz do rzęs świetnie wydłuża.',
        'Jaki podkład polecacie do cery mieszanej?', 'Ten krem BB jest lekki, ale dobrze kryje.',
        'Bardzo lubię ten błyszczyk, nie klei się.', 'Produkt do ust, który utrzymuje się cały dzień.',
        'Kosmetyk idealnie matuje cerę na wiele godzin.', 'Idealny do podkreślenia spojrzenia i wydłużenia rzęs.',
        'Kupiłam wczoraj nowy samochód.', 'Czerwona szminka to klasyk w każdej kosmetyczce.',
        'Ten fluid dobrze się rozprowadza.', 'Pogoda dzisiaj jest okropna.',
        'Nowa maskara z rewolucyjną szczoteczką.'
    ],
    'label': [
        'lipstick', 'mascara', 'foundation', 'foundation', 'lipstick',
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    ]
}
df = pd.DataFrame(all_data)

df_labeled = df.dropna().copy()
df_unlabeled = df[df['label'].isnull()].copy().reset_index(drop=True)

# Wektoryzacja tekstu na podstawie wszystkich dostępnych tekstów
vectorizer = TfidfVectorizer(max_features=1000)
vectorizer.fit(df['text'])

X_labeled = vectorizer.transform(df_labeled['text'])
y_labeled = df_labeled['label']
X_unlabeled = vectorizer.transform(df_unlabeled['text'])

# --- Krok 2: Pętla Samouczenia ---
CONFIDENCE_THRESHOLD = 0.90
model = LogisticRegression(max_iter=200)

print(f"Początkowy rozmiar zbioru treningowego: {len(y_labeled)}")

# Trenowanie modelu bazowego
model.fit(X_labeled, y_labeled)

# Predykcja na nieoznaczonych danych
pred_probas = model.predict_proba(X_unlabeled)
max_probas = np.max(pred_probas, axis=1)

# Selekcja pewnych predykcji
confident_indices = np.where(max_probas >= CONFIDENCE_THRESHOLD)[0]

if len(confident_indices) > 0:
    print(f"Znaleziono {len(confident_indices)} nowych pseudo-etykiet z pewnością >= {CONFIDENCE_THRESHOLD}")
    
    # Generowanie pseudo-etykiet
    pseudo_labels = model.predict(X_unlabeled[confident_indices])
    
    # Wyświetlenie przykładów (dla weryfikacji)
    for i, idx in enumerate(confident_indices):
        print(f"  - Tekst: '{df_unlabeled['text'].iloc[idx]}', Pseudo-etykieta: '{pseudo_labels[i]}'")

    # Powiększenie zbioru treningowego
    X_newly_labeled = X_unlabeled[confident_indices]
    
    X_combined = np.vstack([X_labeled.toarray(), X_newly_labeled.toarray()])
    y_combined = np.concatenate([y_labeled, pseudo_labels])
    
    # Ponowne trenowanie na powiększonym zbiorze
    print("\nPonowne trenowanie na połączonych danych...")
    model.fit(X_combined, y_combined)
    print(f"Finalny rozmiar zbioru treningowego: {len(y_combined)}")
else:
    print("Nie znaleziono predykcji powyżej progu pewności.")

# Najlepsze Praktyki i Kluczowe Ryzyka
# Zacznij od Wysokiego Progu Pewności: To najważniejsza zasada. Lepiej dodać 1000 bardzo pewnych pseudo-etykiet (próg=0.99) niż 10 000 niepewnych (próg=0.80). Zapobiega to najważniejszemu ryzyku: propagacji błędu.
# Propagacja Błędu (Confirmation Bias): Jeśli Twój model jest bardzo pewny swojej błędnej predykcji, dodasz tę błędną etykietę do zbioru treningowego. Model w kolejnej iteracji jeszcze mocniej utwierdzi się w swoim błędzie. Wysoki próg minimalizuje to ryzyko.
# Uważaj na Imbalans Klas: Samouczenie ma tendencję do pogłębiania istniejącego imbalansu. Model jest naturalnie najbardziej pewny co do klas, które już dobrze zna (te z 200 przykładami), a rzadko osiągnie wysoki próg dla klas z 20 przykładami.
# Rozwiązanie: Możesz użyć progów pewności zależnych od klasy lub po każdej iteracji stosować techniki re-balansowania danych (np. oversampling) przed kolejnym treningiem.
# Jakość Modelu Bazowego: Jakość pseudo-etykiet jest bezpośrednio zależna od jakości Twojego pierwszego modelu. Im lepszy model na starcie, tym skuteczniejszy będzie cały proces.
# Strategia i Miejsce w Twoim Projekcie
# Samouczenie najlepiej pasuje jako etap skalowania po słabej superwizji.
# Faza 1 - Fundament: Stwórz solidny zbiór treningowy z Twoich 3k ręcznych etykiet oraz ~20k etykiet o wysokiej jakości wygenerowanych przez Snorkel/kontekstowe LF.
# Faza 2 - Skalowanie (Samouczenie): Wytrenuj na tym 23-tysięcznym zbiorze potężny model (np. BERT). Następnie użyj go w procesie samouczenia, aby wygenerować pseudo-etykiety dla pozostałych ~67k zdań, stosując bardzo wysoki próg pewności.
# Faza 3 - Finalny Trening: Wytrenuj swój ostateczny model na połączonych danych ze wszystkich faz.
# To podejście pozwala bezpiecznie i automatycznie wykorzystać cały potencjał Twojego ogromnego zbioru danych, prowadząc do znacznie bardziej wydajnego i dokładnego modelu końcowego.
