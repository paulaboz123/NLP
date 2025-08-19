# Doskonale. Dotarliśmy do techniki, która nie jest kolejnym modelem klasyfikacyjnym, ale fundamentalnie innym sposobem trenowania, aby model nauczył się tworzyć znacznie lepsze, bardziej odporne i użyteczne reprezentacje (osadzenia) zdań. Mowa o Uczeniu Kontrastywnym (Contrastive Learning).
# To podejście jest idealne dla Twojego problemu z podobnymi etykietami i potencjalnie zaszumionymi danymi ze słabej superwizji.
# Intuicja: Bibliotekarz Organizujący Książki
# Wyobraź sobie, że klasyczny trening (supervised learning) to uczenie bibliotekarza, aby patrząc na okładkę książki, od razu przypisał ją do jednej z 200 półek (etykiet).
# Uczenie kontrastywne to coś zupełnie innego. Nie każesz mu od razu przypisywać książki do półki. Zamiast tego, dajesz mu dwie książki i mówisz:
# "Te dwie książki są o tym samym. Połóż je blisko siebie." (para pozytywna)
# "Te dwie książki są o czymś zupełnie innym. Połóż je daleko od siebie." (para negatywna)
# Po przejrzeniu tysięcy takich par, bibliotekarz sam, bez znajomości nazw półek, zaczyna organizować bibliotekę tak, że wszystkie książki o kryminałach lądują w jednym rogu, a wszystkie o poezji w drugim. Tworzy mapę znaczeń. Dopiero potem, gdy ta mapa jest już gotowa, postawienie etykiet na poszczególnych sekcjach staje się trywialnie proste.
# Dlaczego to jest tak Potężne dla Twojego Projektu?
# Radzenie Sobie z Podobnymi Etykietami: Zmusza model do skupienia się na subtelnych różnicach. Aby odepchnąć od siebie zdania o powder i liquid powder, model musi nauczyć się, że słowo "płyn" (liquid) jest kluczowym wyróżnikiem. Klasyczna klasyfikacja może tego nie wychwycić.
# Odporność na Szum w Etykietach: Nawet jeśli niektóre etykiety ze słabej superwizji są błędne, ogólny "sygnał" z tysięcy par pozytywnych i negatywnych jest znacznie silniejszy. Model uczy się ogólnego "sąsiedztwa" semantycznego, co czyni go mniej podatnym na pojedyncze błędy.
# Lepsze Reprezentacje dla Rzadkich Klas (Few-Shot): Jeśli model nauczy się dobrej mapy znaczeń, to nawet jeśli widział tylko 20 przykładów liquid powder, umieści je we właściwym miejscu (blisko powder, ale w odrębnym skupisku). To sprawia, że klasyfikacja nowych przykładów tej rzadkiej klasy jest znacznie łatwiejsza.
# Jak to Działa w Praktyce? Trening z Użyciem Trójek (Triplet Loss)
# Najpopularniejszą metodą implementacji uczenia kontrastywnego jest tzw. Triplet Loss.
# Tworzenie Trójek: Zamiast podawać modelowi pojedyncze zdania, podczas każdej iteracji treningowej podajesz mu trójkę (triplet):
# Kotwica (Anchor): Dowolne zdanie z Twojego zbioru.
# Pozytyw (Positive): Inne zdanie, które ma tę samą etykietę co Kotwica.
# Negatyw (Negative): Zdanie, które ma inną etykietę niż Kotwica.
# Cel Treningu: Model (np. Sentence-BERT) przekształca każdą z tych trzech zdań w wektor. Celem funkcji straty (Triplet Loss) jest zaktualizowanie wag modelu tak, aby:
# Odległość(Kotwica, Pozytyw) + margines < Odległość(Kotwica, Negatyw)
# Innymi słowy: "Przesuń wektor Pozytywa bliżej Kotwicy, a wektor Negatywa dalej od Kotwicy, i upewnij się, że jest on dalej o co najmniej pewien margines (margin)".
# Wynik: Po tysiącach takich operacji, model tworzy przestrzeń osadzeń, w której zdania o tej samej etykiecie naturalnie grupują się w ciasne klastry, a klastry różnych etykiet są od siebie wyraźnie odseparowane.
# Kod Koncepcyjny: Przygotowanie Danych do Uczenia Kontrastywnego
# Prawdziwa implementacja pętli treningowej jest złożona, ale najważniejsze jest zrozumienie, jak przygotować dane. Biblioteka sentence-transformers ma wbudowane narzędzia, które to ułatwiają.

import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# --- Krok 1: Przygotowanie Danych ---
# Potrzebujemy danych z etykietami. To mogą być Twoje dane oznaczone ręcznie
# lub wysokiej jakości dane ze słabej superwizji.
data = {
    'text': [
        'Ten podkład w płynie idealnie się rozprowadza.',
        'Nowy puder w płynie od tej firmy jest rewelacyjny.',
        'Ta wodoodporna maskara nie osypuje się.',
        'Mój tusz do rzęs świetnie je wydłuża.',
        'Ta szminka ma piękny, matowy kolor.',
        'Bardzo lubię ten błyszczyk, nie klei się.',
        'Puder w kamieniu świetnie matuje cerę.'
    ],
    'label': [
        'liquid_powder', 'liquid_powder', # Para pozytywna
        'mascara', 'mascara', # Para pozytywna
        'lipstick', 'lipstick', # Para pozytywna
        'powder' # Przykład negatywny dla pozostałych
    ]
}
df = pd.DataFrame(data)

# --- Krok 2: Formatowanie Danych dla Sentence-Transformers ---
# Biblioteka potrzebuje danych w określonym formacie.
# Najpierw tworzymy listę przykładów InputExample.
train_examples = []
for index, row in df.iterrows():
    # Mapujemy etykiety tekstowe na liczby całkowite
    # W praktyce zrobisz to dla wszystkich 200 etykiet
    label_map = {'liquid_powder': 0, 'mascara': 1, 'lipstick': 2, 'powder': 3}
    train_examples.append(InputExample(texts=[row['text']], label=label_map[row['label']]))
    
# --- Krok 3: Konfiguracja Modelu i Pętli Treningowej ---

# Załaduj model, który chcesz dostroić
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Specjalny DataLoader, który automatycznie tworzy trójki z podanych danych
# Zapewnia, że każda partia (batch) zawiera przykłady z tych samych klas (do par pozytywnych)
# oraz z różnych klas (do par negatywnych).
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Wybierz funkcję straty kontrastywnej, np. TripletLoss
train_loss = losses.TripletLoss(model=model)

# --- Krok 4: Trenowanie ---
# Model jest trenowany za pomocą metody .fit(), która obsługuje całą pętlę
# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
# (Powyższa linia jest zakomentowana, bo wymaga pełnego środowiska PyTorch)

print("Konfiguracja do uczenia kontrastywnego jest gotowa.")
print("Model, DataLoader i funkcja straty (TripletLoss) zostały zdefiniowane.")
print("Po treningu, model będzie generował znacznie lepsze, bardziej odseparowane osadzenia.")


# Strategia dla Twojego Projektu: Dwustopniowe Fine-tuning
# Nie musisz traktować uczenia kontrastywnego jako finalnego rozwiązania. Najlepiej użyć go jako pośredniego kroku doskonalącego, aby stworzyć potężny "silnik" do rozumienia tekstu.
# Faza 1: Uczenie Kontrastywne (Tworzenie Eksperta od Reprezentacji)
# Weź swój zbiór danych (3k ręcznych + np. 20k najlepszych etykiet ze Snorkela).
# Użyj go do dostrojenia (fine-tune) modelu Sentence-BERT za pomocą Triplet Loss.
# Wynik: Otrzymujesz model, który jest ekspertem w przekształcaniu Twoich zdań o kosmetykach w wektory, które mają sensowną strukturę (podobne rzeczy są blisko, różne daleko). Nazwijmy go Cosmetics-BERT.
# Faza 2: Klasyfikacja (Dodawanie Głowicy Klasyfikacyjnej)
# Teraz weź swój nowy, potężny model Cosmetics-BERT.
# "Zamroź" większość jego warstw (niech już się nie zmieniają) i dodaj na jego końcu głowicę do klasyfikacji wieloetykietowej (tę z 200 neuronami i aktywacją sigmoid, o której mówiliśmy wcześniej).
# Wytrenuj (dostrój) tylko tę ostatnią warstwę na tych samych danych.
# Ponieważ osadzenia dostarczane przez Cosmetics-BERT są już tak dobrze zorganizowane, zadanie finalnego klasyfikatora staje się znacznie, znacznie łatwiejsze. To często prowadzi do skokowego wzrostu dokładności, zwłaszcza na trudnych i rzadkich klasach.
# Zalety (+)	Wady (-)
# Tworzy Wysoce Dyskryminacyjne Reprezentacje: Idealne do odróżniania drobnych niuansów.	Złożoność Implementacji: Wymaga bardziej skomplikowanej pętli treningowej i przygotowania danych w parach/trójkach.
# Odporność na Szum: Uśrednia sygnał z wielu par, co czyni go robustnym.	Wrażliwość na Strategię Próbkowania: Wyniki mogą zależeć od tego, jak "trudne" negatywy są wybierane do trójek.
# Doskonały Krok Wstępny: Idealny do "podgrzania" modelu przed finalną klasyfikacją.	Koszt Obliczeniowy: Trenowanie na trójkach może być wolniejsze niż standardowa klasyfikacja.
