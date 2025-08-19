# Oczywiście. Przechodzimy do Aktywnego Uczenia (Active Learning). Jest to strategia, która stanowi pomost między automatycznym uczeniem maszynowym a niezastąpioną wiedzą ekspercką człowieka. Jej głównym celem nie jest zastąpienie Cię, ale uczynienie Twojej pracy maksymalnie efektywną.
# Jeśli masz ograniczony budżet czasowy na ręczne etykietowanie, Active Learning zapewnia, że każda minuta poświęcona na oznaczenie nowego zdania przynosi jak największą korzyść dla modelu.
# Filozofia: Model jako Ciekawski Uczeń
# Wyobraź sobie swój model jako ucznia, który ma już pewną wiedzę (nauczył się na Twoich 3k+ przykładów), ale wciąż ma wiele wątpliwości.
# Podejście Pasywne (Losowe Etykietowanie): Mówisz uczniowi: "Otwórz książkę na losowej stronie i powiedz mi, czego nie rozumiesz". Uczeń może trafić na coś, co już dobrze zna, albo na coś trywialnego. Twoja pomoc jest wykorzystywana nieefektywnie.
# Podejście Aktywne: Mówisz uczniowi: "Przejrzyj cały materiał i wskaż mi te 50 zagadnień, które sprawiają Ci największy problem i co do których jesteś najbardziej niezdecydowany". Teraz każda Twoja odpowiedź bezpośrednio "łata" największą dziurę w jego wiedzy.
# Active Learning to proces, w którym model sam identyfikuje i "prosi" o etykiety dla danych, które są dla niego najbardziej niejasne, dwuznaczne lub informatywne.
# Cykl Aktywnego Uczenia (The Active Learning Loop)
# Jest to proces iteracyjny, który zazwyczaj przebiega w następujących krokach:
# Trenuj (Train): Wytrenuj model na wszystkich dotychczas dostępnych oznaczonych danych.
# Przewiduj (Predict): Użyj wytrenowanego modelu, aby wygenerować predykcje (wraz z prawdopodobieństwami) dla całej puli nieoznaczonych danych.
# Zapytaj (Query): Zastosuj "strategię zapytania" (query strategy), aby uszeregować wszystkie nieoznaczone zdania od najbardziej "wartościowych" do najmniej.
# Etykietuj (Label): Przedstaw człowiekowi-ekspertowi (Tobie) N najwyżej ocenionych zdań. Ekspert przypisuje im poprawne, wysokiej jakości etykiety.
# Dodaj i Powtórz (Append & Repeat): Dodaj nowo oznaczone dane do zbioru treningowego i rozpocznij cały cykl od nowa.
# Z każdą iteracją model staje się coraz lepszy, ponieważ jest karmiony dokładnie tymi danymi, których najbardziej potrzebuje.
# Strategie Zapytania: Jak Model Wyraża Swoją Niepewność?
# To jest serce Active Learning. Sposób, w jaki wybieramy "najbardziej wartościowe" przykłady. Dla problemów wieloetykietowych, strategie te są nieco bardziej złożone niż w klasyfikacji jednoklasowej.
# Least Confidence (Najmniejsza Pewność):
# Idea: Znajdź zdanie, dla którego najbardziej prawdopodobna etykieta ma najniższą pewność. W przypadku wielu etykiet, można to uśrednić lub wziąć minimum.
# Proste i skuteczne.
# Margin Sampling (Próbkowanie Marginesu):
# Idea: Znajdź zdanie, w którym różnica prawdopodobieństw między "najlepszą" a "drugą najlepszą" etykietą jest najmniejsza. To wskazuje na niezdecydowanie modelu.
# Dobre do rozróżniania podobnych klas.
# Entropy Sampling (Próbkowanie Entropii):
# Idea: Oblicz entropię rozkładu prawdopodobieństw dla każdego zdania. Wysoka entropia oznacza, że prawdopodobieństwa są rozłożone równomiernie (np. [0.3, 0.3, 0.2, 0.2]), co świadczy o maksymalnej niepewności modelu.
# Bardzo solidna, często najlepsza strategia.
# Przewidywana Zmiana Gradientu (Expected Gradient Length - EGL):
# Idea (zaawansowana): Zamiast patrzeć tylko na wyjście modelu, próbujemy oszacować, które zdanie, gdyby zostało oznaczone i dodane do zbioru, spowodowałoby największą zmianę w wagach (gradientach) modelu podczas następnej iteracji treningowej.
# Obliczeniowo drogie, ale teoretycznie najbardziej optymalne.
# Kod: Symulacja Cyklu Aktywnego Uczenia z użyciem modAL
# Ręczne implementowanie pętli AL jest możliwe, ale łatwo o błędy. Biblioteki takie jak modAL (Modular Active Learning for Python) upraszczają ten proces. Zainstaluj: pip install modAL scikit-learn.
# Poniższy kod symuluje proces z użyciem tej biblioteki.

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from modAL.models import ActiveLearner

# --- Krok 1: Przygotowanie Danych ---
# Zaczynamy z bardzo małym, początkowym zbiorem oznaczonym
initial_data = {
    'text': [
        'Ta nowa pomadka ma piękny, matowy kolor.', 'Mój ulubiony tusz do rzęs świetnie wydłuża.',
        'Ten podkład i korektor idealnie kryją niedoskonałości.'
    ],
    'label': [
        'lipstick', 'mascara', 'foundation' # Uproszczenie do jednej etykiety dla modAL
    ]
}
df_initial = pd.DataFrame(initial_data)

# Reszta danych to nasza "pula" nieoznaczonych zdań
pool_data = {
    'text': [
        'Nowa szminka z drobinkami brokatu.',
        'Wodoodporna maskara to must-have na lato.',
        'Fluid i puder w jednym produkcie.',
        'Jaki polecacie krem nawilżający pod makijaż?',
        'Ten błyszczyk pięknie powiększa usta.',
        'Korektor pod oczy, który nie wchodzi w zmarszczki.'
    ],
    # Te etykiety symulują "eksperta" (Ciebie)
    'true_label': [
        'lipstick', 'mascara', 'foundation', 'cream', 'lipstick', 'concealer'
    ]
}
df_pool = pd.DataFrame(pool_data)

# Wektoryzacja (w praktyce użyłbyś osadzeń z BERTa)
vectorizer = TfidfVectorizer()
all_text = pd.concat([df_initial['text'], df_pool['text']])
vectorizer.fit(all_text)

X_initial = vectorizer.transform(df_initial['text'])
y_initial = df_initial['label']
X_pool = vectorizer.transform(df_pool['text'])

# Mapowanie etykiet na liczby
y_initial_numeric = pd.Series(y_initial).astype('category').cat.codes.to_numpy()


# --- Krok 2: Inicjalizacja Active Learnera ---
# `modAL` potrzebuje estymatora zgodnego z `scikit-learn`
estimator = LogisticRegression(max_iter=200)

# Inicjalizacja ActiveLearnera
learner = ActiveLearner(
    estimator=estimator,
    X_training=X_initial.toarray(), y_training=y_initial_numeric
)


# --- Krok 3: Pętla Aktywnego Uczenia ---
N_QUERIES = 3 # Ile razy "zapytamy" eksperta o nową etykietę

for i in range(N_QUERIES):
    print(f"\n===== CYKL AKTYWNEGO UCZENIA: ITERACJA {i+1} =====")
    print(f"Aktualny rozmiar zbioru treningowego: {len(learner.X_training)}")

    # 1. Zapytaj o najbardziej informatywny przykład z puli (domyślna strategia to 'uncertainty_sampling')
    query_idx, query_instance = learner.query(X_pool.toarray())
    
    # 2. Symuluj etykietowanie przez człowieka
    human_label_text = df_pool['true_label'].iloc[query_idx[0]]
    print(f"Zapytanie o zdanie: '{df_pool['text'].iloc[query_idx[0]]}'")
    print(f"Ekspert dostarcza etykietę: '{human_label_text}'")
    
    # Konwersja etykiety tekstowej na numeryczną (w praktyce musisz zarządzać słownikiem etykiet)
    # Dla uproszczenia, mapujemy to ręcznie
    all_labels = np.unique(np.concatenate([y_initial, df_pool['true_label']]))
    label_map = {label: i for i, label in enumerate(all_labels)}
    human_label_numeric = np.array([label_map[human_label_text]])
    
    # 3. Naucz model na nowym przykładzie (dodaje go do zbioru treningowego i retrenuje)
    learner.teach(X=query_instance.reshape(1, -1), y=human_label_numeric)
    
    # 4. Usuń oznaczony przykład z puli
    X_pool = np.delete(X_pool.toarray(), query_idx, axis=0)
    df_pool = df_pool.drop(df_pool.index[query_idx]).reset_index(drop=True)

print(f"\n===== ZAKOŃCZONO PROCES =====")
print(f"Finalny rozmiar zbioru treningowego: {len(learner.X_training)}")

# Uwaga: Standardowa implementacja modAL jest zaprojektowana dla klasyfikacji jednoetykietowej. Adaptacja do problemu wieloetykietowego wymagałaby użycia opakowań (wrapperów) z scikit-multilearn lub stworzenia niestandardowej pętli, ale logika pozostaje ta sama.
# Strategia i Miejsce w Twoim Projekcie
# Kiedy Używać?: Active Learning to idealny ostatni etap optymalizacji Twojego zbioru danych, tuż przed finalnym treningiem. Stosuj go, gdy:
# Masz już solidny model bazowy (wytrenowany na danych z ręcznego etykietowania i słabej superwizji).
# Masz możliwość, ale ograniczoną, dalszego ręcznego etykietowania (np. "mogę poświęcić 8 godzin na oznaczenie kolejnych 500 zdań").
# Chcesz maksymalnie poprawić wydajność modelu, skupiając się na jego najsłabszych punktach.
# Podejście Hybrydowe: Możesz połączyć AL z pseudo-etykietowaniem.
# Użyj pseudo-etykietowania z wysokim progiem, aby automatycznie dodać "łatwe" przykłady.
# Użyj aktywnego uczenia, aby ręcznie oznaczyć "trudne" przykłady.
# Active Learning to potężna metodologia, która zmienia paradygmat z "więcej danych jest zawsze lepsze" na "właściwe dane są znacznie lepsze". W Twoim przypadku, inteligentne wybranie kilkuset dodatkowych zdań do ręcznego oznaczenia może przynieść większy wzrost dokładności niż automatyczne dodanie dziesiątek tysięcy zaszumionych pseudo-etykiet.
