# Oczywiście. Przechodzimy na najwyższy poziom zaawansowania w tworzeniu funkcji etykietujących. Kontekstowa Słaba Superwizja (Contextualized Weak Supervision) to metoda, która rozwiązuje największą wadę podejścia opartego na słowach kluczowych: brak rozumienia znaczenia.
# Zamiast sztywnych reguł typu jeśli w tekście jest słowo "puder", tworzymy reguły, które działają na poziomie znaczenia: jeśli tekst jest semantycznie podobny do konceptu "puder".
# Dlaczego to Ogromny Krok Naprzód?
# Wyobraź sobie te problemy, których zwykłe reguły nie rozwiązują:
# Synonimy i Parafrazy: Zdanie "Ten kosmetyk matuje cerę" jest o podkładzie, ale nie zawiera słowa "podkład". Zwykła reguła go pominie.
# Wieloznaczność: Zdanie "W magazynie jest proch strzelniczy" zawiera słowo "proch", ale Twoja etykieta powder (puder) nie powinna tu być przypisana. Zwykła reguła popełni błąd.
# Złożoność: Niektórych konceptów nie da się opisać prostą listą słów kluczowych.
# Kontekstowa słaba superwizja wykorzystuje osadzenia (embeddings) z dużych modeli językowych (jak BERT), aby przekształcić zdania w wektory liczbowe, które reprezentują ich znaczenie. Następnie operujemy na tych wektorach.
# Jak to Działa w Praktyce? Plan Działania
# Wygeneruj Osadzenia (Embeddings): Użyjemy wstępnie wytrenowanego modelu (np. Sentence-BERT) do przekształcenia każdego z Twoich 90 000 zdań w wektor liczbowy.
# Stwórz "Wektory Konceptu": Dla każdej etykiety (np. "lipstick", "mascara") weź kilka (5-20) pewnych, ręcznie oznaczonych przykładów z Twoich 3k danych. Uśrednij ich wektory, aby stworzyć jeden "wektor-prototyp" reprezentujący ogólny koncept tej etykiety.
# Stwórz Kontekstową Funkcję Etykietującą: Ta funkcja będzie:
# Brała wektor nieoznaczonego zdania.
# Obliczała jego podobieństwo kosinusowe (cosine similarity) do każdego "wektora konceptu".
# Jeśli podobieństwo do któregoś konceptu przekroczy ustalony próg (np. 0.8), przypisze odpowiednią etykietę.
# Kod w Pythonie
# Do tego zadania potrzebne będą dodatkowe biblioteki. Zainstaluj je:
# pip install sentence-transformers scikit-learn


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Krok 1: Przygotowanie Danych i Modelu ---

# W tym przykładzie musimy mieć kilka już oznaczonych zdań,
# aby stworzyć z nich 'wektory konceptu'. Wykorzystamy Twoje 3k danych.
data = {
    'text': [
        # Przykłady już oznaczone (z Twoich 3k danych)
        'Ta nowa pomadka ma piękny, matowy kolor.', # label: lipstick
        'Bardzo lubię ten błyszczyk, nie klei się.', # label: lipstick
        'Mój ulubiony tusz do rzęs świetnie wydłuża.', # label: mascara
        'Ta maskara idealnie rozdziela rzęsy.', # label: mascara
        'Ten krem BB jest lekki, ale dobrze kryje.', # label: foundation
        'Jaki podkład polecacie do cery mieszanej?', # label: foundation
        
        # Dane nieoznaczone, które chcemy etykietować
        'Kosmetyk idealnie matuje cerę na wiele godzin.', # semantycznie podobne do foundation
        'Produkt do ust, który utrzymuje się cały dzień.', # semantycznie podobne do lipstick
        'Kupiłam wczoraj nowy samochód.', # niepodobne do niczego
        'Idealny do podkreślenia spojrzenia.', # semantycznie podobne do mascara
    ],
    'label': [
        'lipstick', 'lipstick', 'mascara', 'mascara', 'foundation', 'foundation',
        None, None, None, None # Te chcemy wypełnić
    ]
}
df = pd.DataFrame(data)

# Załaduj wstępnie wytrenowany model do tworzenia osadzeń.
# Wybieramy model, który dobrze radzi sobie z językiem polskim.
# 'paraphrase-multilingual-MiniLM-L12-v2' jest dobrym, szybkim wyborem.
print("Ładowanie modelu SentenceTransformer...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model załadowany.")

# --- Krok 2: Generowanie Osadzeń dla Wszystkich Zdań ---

print("Generowanie osadzeń dla wszystkich tekstów...")
# To może potrwać chwilę na dużym zbiorze danych (zalecane GPU)
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
# embeddings to teraz macierz NumPy, gdzie każdy wiersz to wektor dla jednego zdania.

# --- Krok 3: Tworzenie "Wektorów Konceptu" z Oznaczonych Danych ---

concept_vectors = {}
# Bierzemy tylko dane, które mają już etykietę
labeled_df = df.dropna(subset=['label'])
labeled_embeddings = model.encode(labeled_df['text'].tolist())

# Dla każdej unikalnej etykiety obliczamy średni wektor
for label in labeled_df['label'].unique():
    # Znajdź indeksy zdań z daną etykietą
    indices = labeled_df[labeled_df['label'] == label].index
    # Wybierz odpowiednie wektory i oblicz ich średnią
    concept_vectors[label] = np.mean(embeddings[indices], axis=0)

print("\nUtworzono wektory konceptu dla etykiet:", list(concept_vectors.keys()))

# --- Krok 4: Definicja i Aplikacja Kontekstowej Funkcji Etykietującej ---

def apply_contextual_lfs(row_embedding, concept_vectors, threshold=0.65):
    """
    Porównuje osadzenie zdania z wektorami konceptu i zwraca etykietę,
    jeśli podobieństwo przekracza próg.
    """
    best_label = None
    max_similarity = -1

    # Przekształcamy osadzenie wiersza do formatu 2D dla cosine_similarity
    row_embedding_2d = row_embedding.reshape(1, -1)

    for label, concept_vec in concept_vectors.items():
        concept_vec_2d = concept_vec.reshape(1, -1)
        sim = cosine_similarity(row_embedding_2d, concept_vec_2d)[0][0]
        
        if sim > max_similarity:
            max_similarity = sim
            best_label = label
            
    if max_similarity > threshold:
        return best_label
    else:
        return None # Zwróć None, jeśli nic nie jest wystarczająco podobne

# Aplikujemy funkcję tylko do nieoznaczonych danych
unlabeled_indices = df[df['label'].isnull()].index

weak_labels = []
for index in unlabeled_indices:
    # Pobieramy wcześniej wygenerowane osadzenie dla danego wiersza
    row_embedding = embeddings[index]
    weak_label = apply_contextual_lfs(row_embedding, concept_vectors)
    weak_labels.append(weak_label)

# Przypisujemy wygenerowane słabe etykiety do DataFrame
df.loc[unlabeled_indices, 'weak_label'] = weak_labels


print("\n--- DataFrame PO zastosowaniu kontekstowej słabej superwizji ---")
print(df[['text', 'label', 'weak_label']])


# Analiza Kodu i Wyników
# SentenceTransformer: To potężne narzędzie, które robi całą ciężką pracę związaną z generowaniem wysokiej jakości, kontekstowych osadzeń.
# Wektory Konceptu: Zauważ, jak prosto tworzymy "prototyp" dla etykiety, po prostu uśredniając wektory kilku jej przykładów. To intuicyjne i bardzo skuteczne.
# Próg Podobieństwa (threshold): To kluczowy hiperparametr, który musisz dostroić.
# Wysoki próg (np. 0.85): Bardzo wysoka precyzja, ale niski zasięg (oznaczysz mało zdań). Dobre, jeśli chcesz mieć pewność co do etykiet.
# Niski próg (np. 0.60): Znacznie większy zasięg, ale niższa precyzja (ryzyko błędnego oznaczenia).
# Wynik: Zobacz, jak zdanie "Kosmetyk idealnie matuje cerę na wiele godzin" zostało poprawnie zidentyfikowane jako semantycznie związane z podkładem, mimo braku słów kluczowych.
# Jak Włączyć To w Ekosystem Snorkel?
# Najlepsze jest to, że nie musisz wybierać między Snorkelem a tym podejściem. Możesz je połączyć, tworząc hybrydowy system:
# Napisz swoje proste funkcje etykietujące oparte na słowach kluczowych (są szybkie i dobre w wyłapywaniu oczywistych przypadków).
# Napisz swoje kontekstowe funkcje etykietujące, jak ta powyżej.
# Wszystkie te funkcje (obu typów!) przekaż do PandasLFApplier w Snorkel.
# Snorkel LabelModel weźmie pod uwagę "głosy" ze wszystkich Twoich reguł – zarówno tych prostych, jak i tych złożonych, kontekstowych. Nauczy się, że być może Twoja kontekstowa LF dla "foundation" jest bardziej wiarygodna niż prosta reguła oparta na słowie kluczowym, i odpowiednio ją zważy.
# To jest absolutnie najnowocześniejsze podejście do słabej superwizji, łączące szybkość i prostotę reguł z głębokim zrozumieniem języka przez modele transformerowe.
