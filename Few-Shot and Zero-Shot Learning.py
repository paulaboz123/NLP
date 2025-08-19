# Doskonale. Dotarliśmy do najbardziej zaawansowanych i, w pewnym sensie, "magicznych" technik, które są stworzone do rozwiązania Twojego najtrudniejszego problemu: etykiet z ekstremalnie małą liczbą przykładów.
# Co zrobić z etykietą liquid powder, która ma tylko 20 przykładów? A co, jeśli w przyszłości pojawi się nowa etykieta, dla której nie masz żadnych przykładów? Na te pytania odpowiadają Few-Shot Learning (FSL) i Zero-Shot Learning (ZSL).
# Obie metody opierają się na jednej, kluczowej idei: zamiast uczyć model rozpoznawania wzorców na podstawie setek przykładów, uczymy go porównywania znaczenia.
# 1. Few-Shot Learning (FSL): Uczenie się z Garstki Przykładów
# FSL jest zaprojektowane dla sytuacji, w których masz bardzo mało danych treningowych dla danej klasy (np. 5, 10, 20 przykładów).
# Intuicja: Detektyw-Ekspert
# Doświadczony detektyw nie musi widzieć 1000 przykładów kradzieży danego typu, aby rozpoznać kolejną. Wystarczy mu zobaczyć 2-3 przypadki, aby zrozumieć koncept i charakterystyczne cechy (modus operandi). FSL działa podobnie – uczy się konceptu etykiety na podstawie kilku przykładów.
# Jak to Działa? Najpopularniejsze Podejście: Sieci Prototypowe (Prototypical Networks)
# To podejście jest niezwykle intuicyjne i skuteczne w NLP.
# Osadzenie (Embedding): Użyj potężnego, wstępnie wytrenowanego modelu (np. Sentence-BERT), aby przekształcić wszystkie zdania (zarówno te nieliczne przykłady, jak i nowe zdania do sklasyfikowania) w wektory w przestrzeni semantycznej.
# Stworzenie Prototypu: Dla każdej rzadkiej klasy (np. liquid powder), weź te 20 zdań, które ją opisują. Przekształć je w wektory, a następnie uśrednij te wektory. Ten średni wektor to "prototyp" – idealny, uśredniony punkt w przestrzeni znaczeniowej, który reprezentuje koncept liquid powder.
# Klasyfikacja przez Podobieństwo: Gdy pojawi się nowe, nieoznaczone zdanie, przekształć je w wektor. Następnie oblicz jego podobieństwo kosinusowe do wszystkich stworzonych prototypów. Etykieta prototypu, który jest najbliżej w przestrzeni znaczeniowej, staje się predykcją.
# Kod Koncepcyjny (z sentence-transformers):


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Załóżmy, że mamy "support set" - nasze nieliczne, oznaczone przykłady
support_data = {
    'text': [
        'Ten podkład w płynie idealnie się rozprowadza.', # liquid_powder
        'Nowy puder w płynie od tej firmy jest rewelacyjny.', # liquid_powder
        'Ta wodoodporna maskara nie osypuje się.', # mascara
        'Mój tusz do rzęs świetnie je wydłuża.', # mascara
    ],
    'label': ['liquid_powder', 'mascara', 'liquid_powder', 'mascara']
}
df_support = pd.DataFrame(support_data)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 1. & 2. Osadzenie i tworzenie prototypów
prototypes = {}
support_embeddings = model.encode(df_support['text'].tolist())

for label in df_support['label'].unique():
    indices = df_support[df_support['label'] == label].index
    # Uśrednianie wektorów dla danej klasy
    prototypes[label] = np.mean(support_embeddings[indices], axis=0)

print("Utworzono prototypy dla:", list(prototypes.keys()))

# 3. Klasyfikacja nowego zdania
query_sentence = "Jaki polecacie fluid do twarzy?"
query_embedding = model.encode([query_sentence])

# Obliczanie podobieństwa do każdego prototypu
similarities = {}
for label, proto_vec in prototypes.items():
    sim = cosine_similarity(query_embedding, proto_vec.reshape(1, -1))
    similarities[label] = sim[0][0]

# Wybór etykiety z najwyższym podobieństwem
prediction = max(similarities, key=similarities.get)

print(f"\nZdanie: '{query_sentence}'")
print(f"Podobieństwa: {similarities}")
print(f"Predykcja: {prediction}")

Zalety (+)	Wady (-)
# Idealne dla Rzadkich Klas: Zaprojektowane specjalnie dla Twojego problemu z etykietami mającymi 20 przykładów.	Wrażliwość na Jakość Przykładów: Jeśli Twoje 20 przykładów jest nietypowych lub zaszumionych, prototyp będzie niedokładny.
# Szybkie Dodawanie Nowych Klas: Jeśli pojawi się nowa klasa z kilkoma przykładami, wystarczy stworzyć dla niej nowy prototyp, bez trenowania całego modelu od zera.	Nie Skaluje się Dobrze: Podejście z prototypami działa świetnie dla kilkudziesięciu klas, ale może być wolne przy setkach i nie pobije modelu BERT wytrenowanego na tysiącach przykładów dla częstych klas.

#         2. Zero-Shot Learning (ZSL): Klasyfikacja Bez Żadnych Przykładów
# To podejście pozwala klasyfikować tekst do etykiet, których model nigdy nie widział w danych treningowych.
# Intuicja: Rozumienie przez Definicję
# Jak wiesz, co to jest "turkusowy eyeliner", nawet jeśli nigdy go nie widziałeś? Bo wiesz, co znaczy słowo "turkusowy" i co znaczy "eyeliner". Twój mózg łączy te dwa koncepty. ZSL działa tak samo – porównuje znaczenie zdania ze znaczeniem samej nazwy etykiety.
# Jak to Działa? Podejście Oparte na Osadzeniach
# Podwójne Osadzenie: Użyj tego samego modelu (np. Sentence-BERT), aby stworzyć wektor dla zdania wejściowego ORAZ wektory dla nazw wszystkich możliwych etykiet (np. dla stringów "pomadka", "tusz do rzęs", "puder w płynie").
# Porównanie w Tej Samej Przestrzeni: Kluczowe jest to, że zarówno znaczenie zdania, jak i znaczenie etykiet lądują w tej samej przestrzeni wektorowej.
# Klasyfikacja przez Podobieństwo: Predykcja to po prostu nazwa etykiety, której wektor jest najbliżej wektora zdania wejściowego. Model dosłownie pyta: "Czy znaczenie tego zdania jest bardziej podobne do znaczenia słowa 'pomadka' czy do znaczenia frazy 'tusz do rzęs'?"
# Kod Koncepcyjny:

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# W ZSL nie potrzebujemy przykładów, tylko listę możliwych etykiet
all_possible_labels = ["pomadka do ust", "tusz do rzęs", "podkład do twarzy", "perfumy", "krem nawilżający"]

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 1. Osadzenie nazw etykiet
label_embeddings = model.encode(all_possible_labels)

# 2. & 3. Klasyfikacja nowego zdania
query_sentence = "Szukam czegoś, co wydłuży moje rzęsy i nada im objętości."
query_embedding = model.encode([query_sentence])

# Obliczanie podobieństwa do każdej etykiety
sim_scores = cosine_similarity(query_embedding, label_embeddings)

# Znalezienie etykiety z najwyższym wynikiem
prediction_index = np.argmax(sim_scores)
prediction = all_possible_labels[prediction_index]

print(f"Zdanie: '{query_sentence}'")
print(f"Predykcja (Zero-Shot): {prediction}")

# Zalety (+)	Wady (-)
# Niesamowita Elastyczność: Możesz dodawać nowe klasy "w locie" bez żadnego retrenowania. Wystarczy dodać nazwę nowej etykiety do listy.	Jakość Zależy od Nazwy Etykiety: Działa dobrze dla opisowych etykiet ("krem nawilżający"). Działa słabo dla niejasnych nazw ("produkt_ogólny_1").
# Brak Potrzeby Danych Treningowych: Rozwiązuje problem zimnego startu dla zupełnie nowych kategorii produktów.	Niższa Dokładność niż FSL/Modele Nadzorowane: To jest bardziej "inteligentne zgadywanie" niż precyzyjna klasyfikacja. Zawsze przegra z modelem, który widział chociaż kilka przykładów.
# Strategia dla Twojego Projektu: Hybrydowe Podejście Kaskadowe
# Nie musisz wybierać jednej metody. Najlepsze systemy łączą je w inteligentny sposób.
# Rdzeń Systemu (dla >50 przykładów): Wytrenuj potężny model wieloetykietowy oparty na BERT na wszystkich klasach, które mają wystarczająco dużo danych. To będzie Twój główny, najdokładniejszy klasyfikator.
# Specjalista FSL (dla 5-50 przykładów): Dla Twoich rzadkich etykiet (jak liquid powder), zbuduj klasyfikator oparty na prototypach (FSL).
# Koło Ratunkowe ZSL (dla 0 przykładów lub niskiej pewności): Zbuduj klasyfikator Zero-Shot jako ostateczność.
# Jak to połączyć w całość?
# Dla nowego zdania:
# Najpierw przepuść je przez główny model BERT.
# Jeśli model zwróci predykcje z wysoką pewnością – super, użyj ich.
# Jeśli model jest niepewny (np. najwyższe prawdopodobieństwo to tylko 30%), przekaż zdanie do klasyfikatora FSL, aby sprawdzić, czy pasuje do którejś z rzadkich klas.
# Jeśli oba modele są niepewne, użyj klasyfikatora ZSL jako ostatecznej próby znalezienia pasującej etykiety lub do zasugerowania potencjalnych, nowych kategorii.
# To podejście daje Ci to, co najlepsze z każdego świata: wysoką dokładność dla częstych klas, rozsądne pokrycie dla rzadkich klas i niesamowitą elastyczność w radzeniu sobie z nowymi, nieznanymi konceptami.
