# Doskonały wybór. Przejście na Snorkel to naturalny i bardzo potężny krok naprzód w porównaniu do ręcznego skryptu. Snorkel formalizuje proces słabej superwizji i, co najważniejsze, inteligentnie agreguje wyniki z Twoich funkcji etykietujących, ucząc się, które z nich są bardziej wiarygodne.
# Dlaczego Snorkel jest Lepszy niż Ręczny Skrypt?
# W naszym poprzednim kodzie, jeśli dwie reguły (np. dla "pomadki" i "podkładu") zostały aktywowane dla tego samego zdania, po prostu łączyliśmy etykiety. Ale co, jeśli jedna reguła jest znacznie dokładniejsza od drugiej? Albo jeśli dwie reguły często się mylą w tych samych przypadkach?
# Snorkel rozwiązuje te problemy. Jego sercem jest Model Etykietujący (Label Model), który:
# Analizuje konflikty i zgodności między Twoimi funkcjami etykietującymi (LFs).
# Uczy się szacowanej dokładności każdej funkcji, nawet bez dostępu do prawdziwych etykiet.
# Waży głosy z LFs i generuje jedną, probabilistyczną, wysokiej jakości etykietę dla każdego punktu danych.
# Kod w Pythonie z użyciem Snorkel
# Najpierw upewnij się, że masz zainstalowaną bibliotekę Snorkel:
# pip install snorkel

import pandas as pd
import re
from snorkel.labeling import labeling_function, PandasLFApplier, LabelingFunction
from snorkel.labeling.model import LabelModel

# --- Krok 1: Przygotowanie danych i zdefiniowanie stałych ---

# Definiujemy stałe dla naszych etykiet. To dobra praktyka.
# 'ABSTAIN' oznacza, że funkcja nie ma zdania na temat danego tekstu.
ABSTAIN = -1
FOUNDATION = 0
LIPSTICK = 1
MASCARA = 2

# Tworzymy ten sam przykładowy DataFrame co poprzednio
data = {
    'text': [
        'Ta nowa pomadka ma piękny, matowy kolor.',
        'Mój ulubiony tusz do rzęs świetnie wydłuża.',
        'Ważne jest, aby dbać o cerę każdego dnia.',
        'Jaki podkład polecacie do cery mieszanej? Musi być trwały.',
        'Ten krem BB jest lekki, ale dobrze kryje.',
        'Bardzo lubię ten błyszczyk, nie klei się.',
        'Promocja na szminki i błyszczyki w naszej drogerii!',
        'Kupiłam wczoraj nowy samochód.',
    ]
}
df = pd.DataFrame(data)

# W Snorkel zazwyczaj pracujemy na całym zbiorze danych (w Twoim przypadku 90k wierszy)
# Model sam nauczy się, które dane są relewantne.
# Dla uproszczenia, w tym przykładzie operujemy na naszym małym df.

# --- Krok 2: Zdefiniowanie Funkcji Etykietujących (LFs) w stylu Snorkel ---

# Używamy dekoratora @labeling_function, aby oznaczyć funkcję jako LF Snorkela.
# Zwraca ona teraz liczbę całkowitą (naszą stałą) lub ABSTAIN.

@labeling_function()
def lf_lipstick_snorkel(x):
    """Etykietuje zdania zawierające słowa kluczowe dla pomadek/szminek."""
    keywords = ['pomadka', 'szminka', 'usta', 'błyszczyk']
    return LIPSTICK if any(re.search(r'\b' + kw + r'\b', x.text, re.IGNORECASE) for kw in keywords) else ABSTAIN

@labeling_function()
def lf_mascara_snorkel(x):
    """Etykietuje zdania zawierające słowa kluczowe dla tuszu do rzęs."""
    keywords = ['tusz do rzęs', 'maskara', 'rzęsy']
    return MASCARA if any(re.search(r'\b' + kw + r'\b', x.text, re.IGNORECASE) for kw in keywords) else ABSTAIN

@labeling_function()
def lf_foundation_snorkel(x):
    """Etykietuje zdania zawierające słowa kluczowe dla podkładu."""
    keywords = ['podkład', 'fluid', 'krem bb', 'krem cc']
    return FOUNDATION if any(re.search(r'\b' + kw + r'\b', x.text, re.IGNORECASE) for kw in keywords) else ABSTAIN

# --- Krok 3: Aplikacja funkcji do danych ---

# Zbieramy wszystkie LFs w listę
lfs = [lf_lipstick_snorkel, lf_mascara_snorkel, lf_foundation_snorkel]

# PandasLFApplier efektywnie aplikuje nasze funkcje do DataFrame
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df) # To tworzy macierz etykiet

print("--- Macierz Etykiet (L_train) wygenerowana przez Snorkel ---")
print("Każdy wiersz to zdanie, każda kolumna to jedna funkcja etykietująca.")
print(L_train)

# --- Krok 4: Trenowanie Modelu Etykietującego i generowanie etykiet ---

# Tworzymy Model Etykietujący. `cardinality` to liczba klas (w naszym przypadku 3).
label_model = LabelModel(cardinality=3, verbose=True)

# Trenujemy model na macierzy etykiet.
# Model uczy się wag i korelacji między LFs, NIE patrząc na tekst!
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

# Generujemy finalne, "oczyszczone" etykiety dla naszych danych
# `tie_break_policy="random"` oznacza, że jeśli model nie może się zdecydować, wybierze losowo.
df['snorkel_label'] = label_model.predict(L=L_train, tie_break_policy="random")


print("\n\n--- DataFrame Z FINALNYMI ETYKIETAMI OD SNORKEL ---")
# Mapujemy liczby z powrotem na nazwy dla czytelności
label_map = {FOUNDATION: 'foundation', LIPSTICK: 'lipstick', MASCARA: 'mascara'}
df['snorkel_label_name'] = df['snorkel_label'].map(label_map)

# Dodajemy kolumnę 'relevance' na podstawie tego, czy Snorkel coś przypisał
df['relevance'] = df['snorkel_label'].apply(lambda x: 0 if x == ABSTAIN else 1)

print(df[['text', 'relevance', 'snorkel_label_name']])


Doskonały wybór. Przejście na Snorkel to naturalny i bardzo potężny krok naprzód w porównaniu do ręcznego skryptu. Snorkel formalizuje proces słabej superwizji i, co najważniejsze, inteligentnie agreguje wyniki z Twoich funkcji etykietujących, ucząc się, które z nich są bardziej wiarygodne.
Dlaczego Snorkel jest Lepszy niż Ręczny Skrypt?
W naszym poprzednim kodzie, jeśli dwie reguły (np. dla "pomadki" i "podkładu") zostały aktywowane dla tego samego zdania, po prostu łączyliśmy etykiety. Ale co, jeśli jedna reguła jest znacznie dokładniejsza od drugiej? Albo jeśli dwie reguły często się mylą w tych samych przypadkach?
Snorkel rozwiązuje te problemy. Jego sercem jest Model Etykietujący (Label Model), który:
Analizuje konflikty i zgodności między Twoimi funkcjami etykietującymi (LFs).
Uczy się szacowanej dokładności każdej funkcji, nawet bez dostępu do prawdziwych etykiet.
Waży głosy z LFs i generuje jedną, probabilistyczną, wysokiej jakości etykietę dla każdego punktu danych.
Kod w Pythonie z użyciem Snorkel
Najpierw upewnij się, że masz zainstalowaną bibliotekę Snorkel:
pip install snorkel
Poniższy kod wykonuje ten sam cel co poprzednio, ale przy użyciu frameworka Snorkel.
code
Python
import pandas as pd
import re
from snorkel.labeling import labeling_function, PandasLFApplier, LabelingFunction
from snorkel.labeling.model import LabelModel

# --- Krok 1: Przygotowanie danych i zdefiniowanie stałych ---

# Definiujemy stałe dla naszych etykiet. To dobra praktyka.
# 'ABSTAIN' oznacza, że funkcja nie ma zdania na temat danego tekstu.
ABSTAIN = -1
FOUNDATION = 0
LIPSTICK = 1
MASCARA = 2

# Tworzymy ten sam przykładowy DataFrame co poprzednio
data = {
    'text': [
        'Ta nowa pomadka ma piękny, matowy kolor.',
        'Mój ulubiony tusz do rzęs świetnie wydłuża.',
        'Ważne jest, aby dbać o cerę każdego dnia.',
        'Jaki podkład polecacie do cery mieszanej? Musi być trwały.',
        'Ten krem BB jest lekki, ale dobrze kryje.',
        'Bardzo lubię ten błyszczyk, nie klei się.',
        'Promocja na szminki i błyszczyki w naszej drogerii!',
        'Kupiłam wczoraj nowy samochód.',
    ]
}
df = pd.DataFrame(data)

# W Snorkel zazwyczaj pracujemy na całym zbiorze danych (w Twoim przypadku 90k wierszy)
# Model sam nauczy się, które dane są relewantne.
# Dla uproszczenia, w tym przykładzie operujemy na naszym małym df.

# --- Krok 2: Zdefiniowanie Funkcji Etykietujących (LFs) w stylu Snorkel ---

# Używamy dekoratora @labeling_function, aby oznaczyć funkcję jako LF Snorkela.
# Zwraca ona teraz liczbę całkowitą (naszą stałą) lub ABSTAIN.

@labeling_function()
def lf_lipstick_snorkel(x):
    """Etykietuje zdania zawierające słowa kluczowe dla pomadek/szminek."""
    keywords = ['pomadka', 'szminka', 'usta', 'błyszczyk']
    return LIPSTICK if any(re.search(r'\b' + kw + r'\b', x.text, re.IGNORECASE) for kw in keywords) else ABSTAIN

@labeling_function()
def lf_mascara_snorkel(x):
    """Etykietuje zdania zawierające słowa kluczowe dla tuszu do rzęs."""
    keywords = ['tusz do rzęs', 'maskara', 'rzęsy']
    return MASCARA if any(re.search(r'\b' + kw + r'\b', x.text, re.IGNORECASE) for kw in keywords) else ABSTAIN

@labeling_function()
def lf_foundation_snorkel(x):
    """Etykietuje zdania zawierające słowa kluczowe dla podkładu."""
    keywords = ['podkład', 'fluid', 'krem bb', 'krem cc']
    return FOUNDATION if any(re.search(r'\b' + kw + r'\b', x.text, re.IGNORECASE) for kw in keywords) else ABSTAIN

# --- Krok 3: Aplikacja funkcji do danych ---

# Zbieramy wszystkie LFs w listę
lfs = [lf_lipstick_snorkel, lf_mascara_snorkel, lf_foundation_snorkel]

# PandasLFApplier efektywnie aplikuje nasze funkcje do DataFrame
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df) # To tworzy macierz etykiet

print("--- Macierz Etykiet (L_train) wygenerowana przez Snorkel ---")
print("Każdy wiersz to zdanie, każda kolumna to jedna funkcja etykietująca.")
print(L_train)

# --- Krok 4: Trenowanie Modelu Etykietującego i generowanie etykiet ---

# Tworzymy Model Etykietujący. `cardinality` to liczba klas (w naszym przypadku 3).
label_model = LabelModel(cardinality=3, verbose=True)

# Trenujemy model na macierzy etykiet.
# Model uczy się wag i korelacji między LFs, NIE patrząc na tekst!
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

# Generujemy finalne, "oczyszczone" etykiety dla naszych danych
# `tie_break_policy="random"` oznacza, że jeśli model nie może się zdecydować, wybierze losowo.
df['snorkel_label'] = label_model.predict(L=L_train, tie_break_policy="random")


print("\n\n--- DataFrame Z FINALNYMI ETYKIETAMI OD SNORKEL ---")
# Mapujemy liczby z powrotem na nazwy dla czytelności
label_map = {FOUNDATION: 'foundation', LIPSTICK: 'lipstick', MASCARA: 'mascara'}
df['snorkel_label_name'] = df['snorkel_label'].map(label_map)

# Dodajemy kolumnę 'relevance' na podstawie tego, czy Snorkel coś przypisał
df['relevance'] = df['snorkel_label'].apply(lambda x: 0 if x == ABSTAIN else 1)

print(df[['text', 'relevance', 'snorkel_label_name']])

# Analiza Wyników i Kluczowe Różnice
# Macierz Etykiet (L_train): Zwróć uwagę na jej wygląd. To tabela, gdzie wiersze to zdania, a kolumny to Twoje funkcje. Wartości to 0 (foundation), 1 (lipstick), 2 (mascara) lub -1 (abstain). Widać tu dokładnie, która funkcja "zagłosowała" na które zdanie.
# LabelModel.fit(): To jest serce operacji. Model analizuje macierz L_train i buduje model probabilistyczny, który estymuje, jak dobra jest każda z Twoich reguł.
# Finalne Etykiety: Kolumna snorkel_label zawiera wynik pracy modelu. Zauważ, że zdania, na które żadna funkcja nie zagłosowała (np. "Kupiłam wczoraj nowy samochód"), otrzymują etykietę -1 (ABSTAIN). Na tej podstawie możesz łatwo ustawić kolumnę relevance.
# Ważna uwaga: Problem wieloetykietowy (Multi-label)
# Standardowy LabelModel w Snorkel jest zaprojektowany do problemów wielo-klasowych, ale jedno-etykietowych (multi-class, single-label), co oznacza, że przypisuje dokładnie jedną etykietę do każdego przykładu.
# Twoje zadanie jest wielo-etykietowe (multi-label), ponieważ jedno zdanie może wspominać o "pomadce i tuszu do rzęs". Jak sobie z tym poradzić w Snorkel?
# Najprostsze i najczęstsze rozwiązanie: Podejście "Jeden kontra Reszta" (One-vs-Rest).
# Zamiast jednego modelu dla 200 etykiet, tworzysz jeden model binarny dla każdej etykiety.
# Dla etykiety "lipstick":
# Twoje LFs zwracają LIPSTICK (np. 1) lub NOT_LIPSTICK (np. 0).
# Trenujesz LabelModel z cardinality=2.
# Wynikiem jest prawdopodobieństwo, czy zdanie dotyczy "lipstick".
# Powtarzasz ten proces dla "mascara", "powder" itd. W ten sposób dla każdego zdania otrzymasz listę prawdopodobieństw (np. lipstick: 0.95, mascara: 0.05, powder: 0.80) i możesz przypisać wszystkie etykiety, które przekroczyły pewien próg.
# Chociaż wymaga to więcej kodu (pętla do trenowania wielu modeli), jest to solidne i sprawdzone podejście do używania Snorkela w problemach wieloetykietowych.
# Podsumowanie i Następne Kroki
# Zacznij od implementacji: Zaimplementuj kod Snorkela dla kilku Twoich najważniejszych lub najczęstszych etykiet, aby zobaczyć, jak działa.
# Analizuj LFs: Snorkel dostarcza narzędzi do analizy Twoich funkcji, np. LFAnalysis, aby zobaczyć ich pokrycie (coverage), konflikty (conflicts) i inne statystyki. To pomoże Ci iteracyjnie poprawiać Twoje reguły.
# Rozważ podejście One-vs-Rest: Jeśli problem wieloetykietowy jest kluczowy, od razu zacznij implementować logikę trenowania osobnych modeli dla każdej etykiety.
# Trenuj model końcowy: Pamiętaj, że wyjście ze Snorkela (kolumna snorkel_label) nie jest końcem pracy. Jest to teraz Twój nowy, znacznie większy i lepszy zbiór treningowy, który możesz użyć do trenowania potężnego modelu, takiego jak BERT, do finalnej klasyfikacji.
