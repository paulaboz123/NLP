import pandas as pd
import re

# --- Krok 1: Stworzenie przykładowego DataFrame ---
# Naśladuje on Twoje dane: 3k oznaczonych i znacznie więcej nieoznaczonych.
# W Twoim przypadku wczytasz tu swój plik, np. df = pd.read_csv('twoje_dane.csv')
data = {
    'text': [
        'Ta nowa pomadka ma piękny, matowy kolor.', # Już oznaczone
        'Mój ulubiony tusz do rzęs świetnie wydłuża.', # Już oznaczone
        'Ważne jest, aby dbać o cerę każdego dnia.', # Nieoznaczone, nierelewantne
        'Jaki podkład polecacie do cery mieszanej? Musi być trwały.', # Nieoznaczone, ale relewantne
        'Ten krem BB jest lekki, ale dobrze kryje.', # Nieoznaczone, potencjalnie relewantne
        'Bardzo lubię ten błyszczyk, nie klei się.', # Nieoznaczone, relewantne
        'Promocja na szminki i błyszczyki w naszej drogerii!', # Nieoznaczone, relewantne (2 etykiety)
        'Kupiłam wczoraj nowy samochód.', # Nieoznaczone, nierelewantne
    ],
    'relevance': [1, 1, 0, 0, 0, 0, 0, 0],
    'label': ['lipstick', 'mascara', None, None, None, None, None, None]
}
df = pd.DataFrame(data)

# --- Krok 2: Zdefiniowanie Funkcji Etykietujących (Labeling Functions) ---
# Każda funkcja przyjmuje tekst i zwraca listę etykiet lub pustą listę.
# Używamy listy, aby obsłużyć przypadki, gdzie jedno zdanie może pasować do wielu etykiet.

def lf_lipstick(text):
    """Etykietuje zdania zawierające słowa kluczowe dla pomadek/szminek."""
    keywords = ['pomadka', 'szminka', 'usta', 'błyszczyk']
    # re.IGNORECASE sprawia, że wielkość liter nie ma znaczenia
    if any(re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE) for keyword in keywords):
        return ['lipstick']
    return []

def lf_mascara(text):
    """Etykietuje zdania zawierające słowa kluczowe dla tuszu do rzęs."""
    keywords = ['tusz do rzęs', 'maskara', 'rzęsy']
    if any(re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE) for keyword in keywords):
        return ['mascara']
    return []

def lf_foundation(text):
    """Etykietuje zdania zawierające słowa kluczowe dla podkładu."""
    keywords = ['podkład', 'fluid', 'krem bb', 'krem cc']
    if any(re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE) for keyword in keywords):
        return ['foundation']
    return []

# Lista wszystkich funkcji etykietujących, które chcemy zastosować
labeling_functions = [
    lf_lipstick,
    lf_mascara,
    lf_foundation
]


# --- Krok 3: Aplikacja funkcji do nieoznaczonych danych ---

print("--- DataFrame PRZED zastosowaniem słabej superwizji ---")
print(df)

# Kopiujemy oryginalną kolumnę label, aby zachować oryginalne etykiety
df['weak_label'] = df['label']

# Przechodzimy tylko przez wiersze, które nie mają jeszcze etykiety
# W Twoim przypadku będzie to `df[df['relevance'] == 0]`
for index, row in df[df['label'].isnull()].iterrows():
    text_to_label = row['text']
    found_labels = []
    
    # Aplikujemy każdą funkcję do tekstu
    for lf in labeling_functions:
        result = lf(text_to_label)
        if result:
            found_labels.extend(result)
            
    # Jeśli znaleziono jakiekolwiek etykiety, aktualizujemy wiersz
    if found_labels:
        # Usuwamy duplikaty, jeśli jakaś reguła odpaliła się podwójnie
        unique_labels = list(set(found_labels))
        
        # Aktualizujemy dane w DataFrame
        df.loc[index, 'weak_label'] = ', '.join(unique_labels) # Łączymy etykiety stringiem
        df.loc[index, 'relevance'] = 1 # Oznaczamy jako relewantne

print("\n\n--- DataFrame PO zastosowaniu słabej superwizji ---")
print(df)



# Kluczowe Elementy Kodu i Wyjaśnienie
# Przykładowy DataFrame: Tworzy mały zbiór danych, który odzwierciedla Twój problem. Masz dane już oznaczone (relevance = 1) i dane czekające na analizę (relevance = 0, label = None).
# Funkcje Etykietujące (LFs):
# Są to proste funkcje w Pythonie. Każda z nich "specjalizuje się" w wyszukiwaniu jednej kategorii etykiet.
# Użyłem re.search(r'\b' + keyword + r'\b', ...) zamiast prostego keyword in text. \b oznacza granicę słowa, co zapobiega fałszywym dopasowaniom (np. słowo "maska" nie zostanie dopasowane do "maskara").
# Każda funkcja zwraca listę, co jest kluczowe dla problemu z wieloma etykietami. Dzięki temu zdanie "Promocja na szminki i błyszczyki" może otrzymać etykietę ['lipstick'] od funkcji lf_lipstick.
# Główna Pętla Aplikująca:
# df[df['label'].isnull()].iterrows(): Iterujemy tylko po tych wierszach, które potrzebują etykiety. To znacznie przyspiesza proces na dużym zbiorze danych.
# found_labels.extend(result): Zbieramy wszystkie etykiety zwrócone przez wszystkie funkcje. Jeśli zdanie pasuje do reguł dla podkładu i tuszu, found_labels będzie zawierać obie te etykiety.
# df.loc[index, 'weak_label'] = ... i df.loc[index, 'relevance'] = 1: Jeśli znaleźliśmy co najmniej jedną etykietę, aktualizujemy nasz DataFrame. Zmieniamy relevance na 1 i wpisujemy znalezione "słabe etykiety". Zdecydowałem się połączyć je w jeden string oddzielony przecinkiem, co jest częstą praktyką.
# Jak Dostosować i Rozbudować Ten Kod?
# Dodaj Więcej Funkcji: Dla Twoich 200 etykiet, musisz stworzyć znacznie więcej funkcji. Pomyśl o synonimach i różnych sformułowaniach. Jedna etykieta może mieć kilka funkcji (np. jedna oparta na słowach kluczowych, druga na bardziej złożonym wzorcu).
# Użyj Wyrażeń Regularnych (Regex): Dla bardziej skomplikowanych wzorców, wyrażenia regularne są niezwykle potężne. Np. do wyszukiwania pojemności produktu ("(\d+)\s?ml").
# Negatywne Reguły: Możesz tworzyć funkcje, które oznaczają coś jako definitywnie nierelewantne. Jeśli zdanie zawiera "samochód" lub "polityka", możesz oznaczyć je jako relevance = 0 z większą pewnością.
# Zarządzanie Konfliktami: Co jeśli jedna reguła mówi "lipstick", a druga "powder"? W tym prostym podejściu przypisujemy obie. Bardziej zaawansowane frameworki jak Snorkel używają modelu generatywnego do uczenia się, która reguła jest bardziej wiarygodna i rozwiązują takie konflikty probabilistycznie. Na razie jednak przypisanie obu jest dobrym startem.
# Zalety i Wady Tego Podejścia
# Zalety:
# Błyskawiczne Skalowanie: Możesz wygenerować tysiące nowych etykiet w kilka minut, zamiast spędzać tygodnie na ręcznym oznaczaniu.
# Wykorzystanie Wiedzy Domenowej: Nikt nie zna tych danych lepiej niż Ty. Możesz przełożyć swoją wiedzę na konkretne, działające reguły.
# Pełna Kontrola i Interpretowalność: Wiesz dokładnie, dlaczego dane zdanie otrzymało daną etykietę.
# Wady:
# Generowanie Szumu: Reguły nie są idealne. Mogą pojawić się błędy (np. zdanie "Moja córka maluje usta lalki" zostanie oznaczone jako 'lipstick', choć kontekst jest inny). Dlatego nazywamy te etykiety "słabymi".
# Ograniczony Zasięg (Low Recall): Twoje reguły nie wychwycą wszystkich relewantnych zdań, tylko te, które pasują do wzorców.
# Wymaga Ręcznej Pracy: Stworzenie dobrych reguł dla 200 etykiet to wciąż sporo pracy.
# Mimo wad, ten zbiór danych (oryginalne 3k etykiet + nowo wygenerowane "słabe" etykiety) będzie znacznie lepszym punktem startowym do trenowania Twojego modelu klasyfikacji niż tylko oryginalne 3k przykładów.
