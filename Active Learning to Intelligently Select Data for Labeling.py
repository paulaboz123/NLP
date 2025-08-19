# Doskonale. Przechodzimy teraz do Aktywnego Uczenia (Active Learning). Jest to strategia, która fundamentalnie różni się od słabej superwizji i pseudo-etykietowania. Tamte metody starały się zautomatyzować proces etykietowania. Active Learning natomiast ma na celu uczynić pracę człowieka-eksperta (czyli Twoją) tak wydajną, jak to tylko możliwe.
# Filozofia: Nie Pracuj Ciężko, Pracuj Mądrze
# Wyobraź sobie, że masz czas na ręczne oznaczenie tylko 100 kolejnych zdań z Twojej puli 87 000. Które wybierzesz?
# Podejście losowe: Wybierasz 100 losowych zdań. Może trafisz na coś ciekawego, a może na 100 oczywistych lub nierelewantnych przykładów, które niczego nowego nie nauczą Twojego modelu.
# Podejście Aktywnego Uczenia: Pytasz swój aktualny model: "Których 100 zdań z tej puli rozumiesz najsłabiej? Pokaż mi je, a ja powiem Ci, jakie są prawidłowe etykiety".
# Active Learning to proces, w którym model aktywnie prosi człowieka o pomoc w przypadkach, które są dla niego najbardziej niejasne lub dwuznaczne.
# Cykl Aktywnego Uczenia (The Active Learning Loop)
# To jest iteracyjny proces, który wygląda następująco:
# Trenuj: Wytrenuj model na wszystkich dotychczas oznaczonych danych.
# Przewiduj: Użyj modelu, aby przewidzieć prawdopodobieństwa dla każdej klasy na całym nieoznaczonym zbiorze danych (tzw. "puli").
# Zapytaj (Query): Zastosuj "strategię zapytania", aby obliczyć, które zdania są najbardziej "informatywne" (czyli te, co do których model jest najbardziej niepewny).
# Etykietuj: Zaprezentuj N najbardziej informatywnych zdań człowiekowi, który przypisuje im poprawne, wysokiej jakości etykiety.
# Dodaj i Powtórz: Dodaj nowo oznaczone dane do zbioru treningowego i zacznij cykl od nowa.
# Każda iteracja "łata" dziury w wiedzy modelu, skupiając się na najtrudniejszych dla niego przypadkach.
# Strategie Zapytania (Query Strategies): Jak Mierzyć Niepewność?
# To jest serce Active Learning. Najpopularniejsze strategie to:
# Least Confidence Sampling (Najmniejsza Pewność):
# Idea: Wybierz zdanie, dla którego najbardziej prawdopodobna etykieta ma najniższe prawdopodobieństwo.
# Przykład: Model przewiduje Zdanie A -> [lipstick: 0.45, mascara: 0.30, powder: 0.25]. Najwyższe prawdopodobieństwo to tylko 45%. Model jest bardzo niepewny. To jest dobry kandydat.
# Proste i bardzo skuteczne.
# Margin Sampling (Próbkowanie Marginesu):
# Idea: Wybierz zdanie, w którym różnica (margines) między dwiema najbardziej prawdopodobnymi etykietami jest najmniejsza.
# Przykład: Model przewiduje Zdanie B -> [lipstick: 0.51, mascara: 0.48, powder: 0.01]. Model jest "rozdarty" między dwiema opcjami. Oznaczenie tego zdania pomoże mu najlepiej nauczyć się granicy decyzyjnej między nimi.
# Entropy Sampling (Próbkowanie Entropii):
# Idea: Wybierz zdanie, dla którego rozkład prawdopodobieństw jest najbardziej "chaotyczny" (ma najwyższą entropię).
# Przykład: Model przewiduje Zdanie C -> [lipstick: 0.33, mascara: 0.33, powder: 0.34]. Prawdopodobieństwa są prawie równomiernie rozłożone, co oznacza maksymalną niepewność modelu.
# Kod: Symulacja Cyklu Aktywnego Uczenia
# Poniższy kod symuluje ten proces. Użyjemy strategii "Least Confidence", ponieważ jest najłatwiejsza do zaimplementowania i zrozumienia.


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# !pip install modAL - opcjonalnie, dla bardziej zaawansowanych zastosowań
# W tym przykładzie zbudujemy logikę ręcznie dla celów edukacyjnych.

# --- Krok 1: Przygotowanie Danych ---
# Zaczynamy z bardzo małym zbiorem oznaczonym
labeled_data = {
    'text': ['Ta nowa pomadka ma piękny, matowy kolor.', 'Mój ulubiony tusz do rzęs świetnie wydłuża.'],
    'label': ['lipstick', 'mascara']
}
df_labeled = pd.DataFrame(labeled_data)

# Reszta danych to nasza "pula" nieoznaczonych zdań, z której będziemy wybierać
pool_data = {
    'text': [
        'Jaki podkład polecacie do cery mieszanej?',
        'Ten krem BB jest lekki, ale dobrze kryje.',
        'Bardzo lubię ten błyszczyk, nie klei się.',
        'Produkt do ust, który utrzymuje się cały dzień.',
        'Kosmetyk idealnie matuje cerę na wiele godzin.',
        'Idealny do podkreślenia spojrzenia i wydłużenia rzęs.',
        'Czerwona szminka to klasyk w każdej kosmetyczce.'
    ],
    # Te etykiety symulują "eksperta" (Ciebie), który zna odpowiedzi
    'true_label': ['foundation', 'foundation', 'lipstick', 'lipstick', 'foundation', 'mascara', 'lipstick']
}
df_pool = pd.DataFrame(pool_data)

# --- Początkowa konfiguracja ---
vectorizer = TfidfVectorizer()
# Uczymy wektoryzatora na wszystkich tekstach, aby słownictwo było spójne
all_text = pd.concat([df_labeled['text'], df_pool['text']])
vectorizer.fit(all_text)

X_labeled = vectorizer.transform(df_labeled['text'])
y_labeled = df_labeled['label']
X_pool = vectorizer.transform(df_pool['text'])

model = LogisticRegression(max_iter=200)

# --- Krok 2: Pętla Aktywnego Uczenia ---
N_QUERIES = 3  # Ile zdań chcemy oznaczyć w sumie
BATCH_SIZE = 1 # Ile zdań oznaczamy w jednej iteracji (w praktyce może być > 1)

for i in range(N_QUERIES):
    print(f"\n===== CYKL AKTYWNEGO UCZENIA: ITERACJA {i+1} =====")
    
    # 1. Trenuj model
    print(f"Trenowanie na {X_labeled.shape[0]} przykładach...")
    model.fit(X_labeled, y_labeled)
    
    # 2. Przewiduj na puli
    pred_probas = model.predict_proba(X_pool)
    
    # 3. Zastosuj strategię zapytania (Least Confidence)
    confidence_scores = np.max(pred_probas, axis=1)
    
    # Sortujemy indeksy od najmniej pewnych do najbardziej pewnych
    uncertainty_ranking = np.argsort(confidence_scores) 
    
    # Wybieramy BATCH_SIZE najbardziej niepewnych
    query_indices = uncertainty_ranking[:BATCH_SIZE]
    
    # 4. Symuluj etykietowanie przez człowieka
    for idx in query_indices:
        text_to_label = df_pool['text'].iloc[idx]
        human_label = df_pool['true_label'].iloc[idx]
        print(f"Zapytanie do eksperta: '{text_to_label}' -> Ekspert odpowiada: '{human_label}'")
        
        # 5. Dodaj nowe dane do zbioru treningowego
        new_X = X_pool[idx]
        new_y = np.array([human_label])
        
        X_labeled = np.vstack([X_labeled.toarray(), new_X.toarray()])
        y_labeled = np.concatenate([y_labeled, new_y])
    
    # 6. Usuń oznaczone dane z puli
    X_pool = np.delete(X_pool.toarray(), query_indices, axis=0)
    df_pool = df_pool.drop(df_pool.index[query_indices]).reset_index(drop=True)

print(f"\n===== ZAKOŃCZONO PROCES =====")
print(f"Finalny rozmiar zbioru treningowego: {X_labeled.shape[0]}")```

### Strategia dla Twojego Projektu i Najlepsze Praktyki

1.  **Kiedy Używać?**: Active Learning jest najbardziej wartościowe, gdy:
    *   Masz już działający model bazowy (np. po słabej superwizji).
    *   Masz możliwość dalszego, ale **ograniczonego** ręcznego etykietowania.
    *   Chcesz "dostroić" model i poprawić jego działanie na trudnych, granicznych przypadkach.

2.  **Etykietowanie Partiami (Batch Mode)**: Pytanie o jedno zdanie na raz jest nieefektywne. W praktyce ustawia się `BATCH_SIZE` na większą liczbę (np. 50-200). Wybierasz `N` najbardziej niepewnych przykładów, etykietujesz je wszystkie, dodajesz do zbioru i dopiero wtedy trenujesz model od nowa.

3.  **Użyj Dedykowanych Bibliotek**: Ręczne budowanie pętli jest dobre do nauki. W produkcji warto użyć bibliotek takich jak **`modAL`** (Modular Active Learning for Python), które mają zaimplementowane różne strategie zapytań i upraszczają cały proces.

4.  **Problem "Zimnego Startu"**: Active Learning nie działa dobrze na samym początku, gdy model nie wie prawie nic. Dlatego tak ważne jest, aby zacząć od przyzwoitego zbioru danych (np. Twoje 3k etykiet + wyniki ze Snorkela).

### Podsumowanie

# | Zalety (+) | Wady (-) |
# | :--- | :--- |
# | **Maksymalna Wydajność Etykietowania**: Każda oznaczona próbka ma duży wpływ na poprawę modelu. | **Wymaga Człowieka w Pętli**: Proces nie jest w pełni automatyczny i jest tak szybki, jak szybko etykietuje ekspert. |
# | **Szybka Poprawa Modelu**: Może prowadzić do znacznego wzrostu dokładności przy znacznie mniejszej liczbie etykiet w porównaniu do etykietowania losowego. | **Koszt Obliczeniowy**: W każdej iteracji trzeba wykonać predykcję na całej dużej puli nieoznaczonych danych. |
# | **Skupienie na Granicach Decyzyjnych**: Naturalnie znajduje najtrudniejsze przykłady, wzmacniając model tam, gdzie jest to najbardziej potrzebne. | **Może być podatny na outliery**: Czasem model jest niepewny co do dziwnych, nietypowych danych, które nie są reprezentatywne. |

# Dla Ciebie Active Learning to idealne narzędzie do **finalnego doszlifowania** Twojego zbioru danych. Po masowym wygenerowaniu etykiet za pomocą słabej superwizji i pseudo-etykietowania, możesz użyć Active Learning, aby inteligentnie wybrać kilkaset "złotych" przykładów do ręcznego oznaczenia, co znacząco podniesie jakość Twojego finalnego modelu.

