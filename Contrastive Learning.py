# Oczywiście. Przechodzimy do Uczenia Kontrastywnego (Contrastive Learning). Jest to jedna z najpotężniejszych technik samonadzorowanego i pół-nadzorowanego uczenia, która zrewolucjonizowała w ostatnich latach przetwarzanie obrazów i języka naturalnego.
# W przeciwieństwie do poprzednich metod, uczenie kontrastywne to nie jest konkretna architektura modelu, ale strategia treningowa. Jej celem nie jest bezpośrednia klasyfikacja, ale nauczenie modelu tworzenia lepszych, bardziej znaczących i użytecznych reprezentacji (osadzeń) danych. Dobrze zorganizowane osadzenia sprawiają, że finalne zadanie klasyfikacji staje się znacznie łatwiejsze.
# Intuicja: Organizowanie Zdjęć z Wakacji
# Wyobraź sobie, że masz tysiące nieposortowanych zdjęć z wakacji.
# Klasyczny trening (Supervised): Pokazujesz modelowi zdjęcie i mówisz "to jest plaża", "to jest las", "to jest zabytek". Model uczy się przypisywać etykiety.
# Uczenie Kontrastywne (Contrastive): Pokazujesz modelowi trzy zdjęcia:
# Zdjęcie wschodu słońca na plaży (Kotwica).
# Inne zdjęcie tej samej plaży, ale zrobione w południe (Pozytyw).
# Zdjęcie z wycieczki w góry (Negatyw).
# Dajesz modelowi jedno proste polecenie: "Zmień swoje postrzeganie tak, aby zdjęcie 1 i 2 były dla Ciebie jak najbardziej podobne, a zdjęcie 1 i 3 jak najbardziej różne."
# Po powtórzeniu tego procesu tysiące razy, model sam, bez znajomości etykiet, zacznie organizować zdjęcia. Wszystkie zdjęcia plaży wylądują w jednym "semantycznym" rogu, wszystkie górskie w innym, a te z miasta jeszcze gdzie indziej. Model nauczy się mapy wizualnych podobieństw.
# Jak to Działa w NLP?
# W przetwarzaniu języka naturalnego działamy analogicznie, używając zdań zamiast obrazów.
# Kluczowe Składniki:
# Enkoder (Encoder): To model (np. BERT, RoBERTa), który przekształca zdanie w wektor liczbowy (osadzenie). Uczenie kontrastywne ma na celu "wytrenowanie" tego enkodera.
# Pary Pozytywne (Positive Pairs): Dwa zdania, które powinny mieć podobne reprezentacje. Mogą to być:
# To samo zdanie, ale poddane augmentacji (np. z usuniętymi słowami, zmienionym szykiem, przetłumaczone w tę i z powrotem) – to jest podejście samonadzorowane (self-supervised), jak w popularnym modelu SimCSE.
# Dwa różne zdania, ale mające tę samą etykietę w Twoim zbiorze danych (np. dwa różne zdania o lipstick) – to jest podejście nadzorowane (supervised).
# Pary Negatywne (Negative Pairs): Dwa zdania, które powinny mieć różne reprezentacje. Zazwyczaj są to losowe zdania z tej samej partii danych (in-batch negatives), które nie są parą pozytywną.
# Funkcja Straty (InfoNCE Loss):
# Sercem uczenia kontrastywnego jest funkcja straty, taka jak InfoNCE (Noise Contrastive Estimation). Jej celem jest maksymalizacja podobieństwa (np. kosinusowego) między wektorami pary pozytywnej, jednocześnie minimalizując podobieństwo do wszystkich par negatywnych w danej partii.
# Loss = -log [ sim(z_i, z_j) / (sim(z_i, z_j) + Σ sim(z_i, z_k)) ]
# Gdzie:
# z_i, z_j to osadzenia pary pozytywnej.
# z_k to osadzenia wszystkich par negatywnych.
# Mówiąc prościej, strata jest niska, gdy podobieństwo pary pozytywnej jest wysokie, a podobieństwo do wszystkich negatywów jest niskie.
# Dlaczego to jest idealne dla Twojego projektu?
# Tworzenie Dyskryminacyjnych Reprezentacji: Zmusza model do skupienia się na subtelnych różnicach, które odróżniają powder od liquid powder. Aby odepchnąć te dwa koncepty od siebie w przestrzeni wektorowej, model musi nauczyć się, że cecha "płynności" jest kluczowa.
# Wykorzystanie Nieoznaczonych Danych: W wersji samonadzorowanej (np. SimCSE), możesz użyć wszystkich swoich 90 000 zdań do treningu, a nie tylko tych 3k oznaczonych! Model uczy się ogólnej struktury języka w Twojej domenie (kosmetyki), co jest ogromną zaletą.
# Odporność na Szum: Podobnie jak w przypadku Triplet Loss, model uczy się na ogólnym sygnale z tysięcy par, co czyni go mniej wrażliwym na pojedyncze błędne etykiety w podejściu nadzorowanym.
# Kod Koncepcyjny: Nadzorowane Uczenie Kontrastywne

import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from collections import defaultdict

# --- Krok 1: Przygotowanie Danych w Parach ---
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
        'liquid_powder', 'liquid_powder',
        'mascara', 'mascara',
        'lipstick', 'lipstick',
        'powder'
    ]
}
df = pd.DataFrame(data)

# Musimy stworzyć listę przykładów InputExample, ale inaczej niż poprzednio.
# Dla wielu funkcji straty kontrastywnej, wystarczy podać tekst i etykietę.
train_examples = []
for index, row in df.iterrows():
    # Mapujemy etykiety tekstowe na liczby całkowite
    label_map = {'liquid_powder': 0, 'mascara': 1, 'lipstick': 2, 'powder': 3}
    train_examples.append(InputExample(texts=[row['text']], label=label_map[row['label']]))

# --- Krok 2: Konfiguracja Modelu i Straty ---
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Specjalny DataLoader, który grupuje zdania o tej samej etykiecie w jednej partii.
# Jest to kluczowe, aby model mógł tworzyć pary pozytywne wewnątrz partii.
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)

# Wybieramy funkcję straty, która jest przeznaczona do nadzorowanego uczenia kontrastywnego.
# ContrastiveLoss, OnlineContrastiveLoss, TripletLoss - to popularne wybory.
# W nowszych wersjach biblioteki, można użyć dedykowanych loaderów i strat,
# np. MultipleNegativesRankingLoss, która jest bardzo wydajna.
train_loss = losses.BatchAllTripletLoss(model=model)

# --- Krok 3: Trenowanie ---
# Model jest trenowany za pomocą metody .fit(), która obsługuje całą pętlę.
# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
# (Powyższa linia jest zakomentowana, bo wymaga pełnego środowiska PyTorch)

print("Konfiguracja do nadzorowanego uczenia kontrastywnego jest gotowa.")
print("Model, specjalny DataLoader i funkcja straty (BatchAllTripletLoss) zostały zdefiniowane.")
print("Po treningu, model będzie generował osadzenia, gdzie zdania z tą samą etykietą będą bardzo blisko siebie.")


# Strategia: Uczenie Kontrastywne jako Krok Wstępny (Pre-training)
# Najpotężniejszym sposobem wykorzystania tej techniki jest potraktowanie jej jako pośredniego, specyficznego dla domeny etapu pre-treningu.
# Faza 1: Samonadzorowany Pre-training na Wszystkich Danych
# Weź wszystkie swoje 90 000 zdań (oznaczone i nieoznaczone).
# Zastosuj samonadzorowaną metodę kontrastywną (np. SimCSE). Polega ona na tym, że dla każdego zdania tworzysz parę pozytywną, przepuszczając je przez enkoder dwa razy z włączonym dropoutem. Dropout wprowadza drobny "szum", więc model uczy się, że te dwie lekko różne wersje powinny być blisko siebie.
# Wynik: Otrzymujesz model, który doskonale rozumie ogólną semantykę języka używanego w Twojej domenie kosmetyków.
# Faza 2: Nadzorowany Pre-training na Oznaczonych Danych
# Weź model z Fazy 1.
# Teraz przeprowadź nadzorowane uczenie kontrastywne (jak w kodzie powyżej), używając Twoich 3k+ oznaczonych zdań. To "dostroi" już dobrze zorganizowaną przestrzeń wektorową, przyciągając do siebie przykłady z tych samych klas.
# Wynik: Otrzymujesz model-eksperta, który generuje niezwykle dobrze zorganizowane osadzenia.
# Faza 3: Finalna Klasyfikacja
# Weź model z Fazy 2. "Zamroź" go i dodaj na końcu prostą głowicę klasyfikacyjną.
# Wytrenuj tę głowicę na oznaczonych danych. Ponieważ osadzenia są tak dobre, ten ostatni etap jest bardzo szybki i skuteczny.
# To podejście jest absolutnym state-of-the-art i chociaż jest najbardziej złożone w implementacji, daje największe szanse na osiągnięcie najwyższej dokładności, zwłaszcza w tak trudnym i drobnoziarnistym problemie jak Twój.
# Poniższy kod pokazuje, jak można użyć biblioteki sentence-transformers do przeprowadzenia nadzorowanego uczenia kontrastywnego. Używa ona par zdań z tą samą etykietą jako par pozytywnych.
