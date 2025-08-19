# Oczywiście. Skoncentrujmy się dogłębnie na Uczeniu Pół-nadzorowanym (Semi-Supervised Learning), a w szczególności na jego najpopularniejszej technice: Samouczeniu (Self-Training) poprzez Pseudo-etykietowanie (Pseudo-Labeling).
# To jedna z najbardziej pragmatycznych i skutecznych metod na wykorzystanie Twoich 87 000 nieoznaczonych zdań w pełni automatyczny sposób.
# Intuicja: Student z kluczem odpowiedzi
# Wyobraź sobie studenta, który ma do rozwiązania 3 000 zadań z kluczem odpowiedzi (Twoje oznaczone dane) i 87 000 zadań bez odpowiedzi (Twoje nieoznaczone dane).
# Student najpierw uczy się solidnie na 3 000 zadań z odpowiedziami.
# Następnie próbuje rozwiązać pozostałe 87 000 zadań.
# Po rozwiązaniu, wraca do tych zadań, przy których czuje się absolutnie pewny swojego rozwiązania (np. na 99% jest pewien, że odpowiedź to "A").
# Traktuje te pewne, samodzielnie uzyskane odpowiedzi tak, jakby były w oficjalnym kluczu i dopisuje je do materiału do nauki.
# Teraz uczy się ponownie, ale już z większej puli "prawidłowych" zadań.
# Tym studentem jest Twój model, a te pewne odpowiedzi to pseudo-etykiety.
# Kluczowa Idea i Proces Krok po Kroku
# Celem jest wykorzystanie własnych predykcji modelu do iteracyjnego powiększania zbioru treningowego. Proces jest prosty, ale diabeł tkwi w szczegółach (głównie w progu pewności).
# Trenowanie Wstępne: Wytrenuj swój najlepszy możliwy model (np. BERT, Logistic Regression na dobrych cechach) na zbiorze oznaczonym (Twoje 3k danych + ewentualne etykiety ze słabej superwizji).
# Predykcja na Nieoznaczonych Danych: Użyj tego modelu, aby przewidzieć prawdopodobieństwa dla każdej z 200 klas dla wszystkich Twoich nieoznaczonych zdań.
# Filtrowanie i Selekcja: Ustal próg pewności (confidence threshold), np. 0.95. Znajdź wszystkie te zdania, dla których model przypisał jakąś etykietę z prawdopodobieństwem wyższym niż ten próg.
# Tworzenie Pseudo-Etykiet: Przypisz wybranym zdaniom te etykiety, których model był tak pewny. To są Twoje "pseudo-etykiety".
# Powiększenie Zbioru: Dodaj zdania z pseudo-etykietami do swojego oryginalnego zbioru treningowego.
# Ponowne Trenowanie: Wytrenuj model od nowa (lub kontynuuj trenowanie) na tym nowym, powiększonym zbiorze danych.
# (Opcjonalnie) Iteracja: Możesz powtórzyć kroki 2-6 kilka razy, stopniowo obniżając próg pewności lub po prostu dodając coraz więcej pseudo-etykiet w kolejnych rundach.
# Kod: Bardziej Szczegółowy Przykład
# Poniższy kod demonstruje iteracyjny proces samouczenia. Jest to rozwinięcie poprzedniego przykładu, aby lepiej pokazać, jak to działa w praktyce.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- Krok 1: Przygotowanie Danych ---
# W praktyce, df_labeled to Twoje 3k+ danych, a df_unlabeled to reszta.
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
df_unlabeled = df[df['label'].isnull()].copy()

# Wektoryzacja tekstu
vectorizer = TfidfVectorizer(max_features=1000) # Ograniczamy liczbę cech
vectorizer.fit(df['text'])

X_labeled = vectorizer.transform(df_labeled['text'])
y_labeled = df_labeled['label']
X_unlabeled = vectorizer.transform(df_unlabeled['text'])

# --- Krok 2: Pętla Samouczenia (Self-Training Loop) ---
N_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.85

model = LogisticRegression(max_iter=200)

for i in range(N_ITERATIONS):
    print(f"\n===== ITERACJA {i+1} =====")
    
    # Trenowanie modelu na aktualnie dostępnych danych oznaczonych
    print(f"Trenowanie na {len(y_labeled)} przykładach...")
    model.fit(X_labeled, y_labeled)
    
    # Sprawdzenie, czy są jeszcze dane do etykietowania
    if X_unlabeled.shape[0] == 0:
        print("Brak nieoznaczonych danych. Koniec pętli.")
        break
        
    # Predykcja prawdopodobieństw na danych nieoznaczonych
    pred_probas = model.predict_proba(X_unlabeled)
    max_probas = np.max(pred_probas, axis=1)
    
    # Selekcja pewnych predykcji
    confident_indices = np.where(max_probas >= CONFIDENCE_THRESHOLD)[0]
    
    if len(confident_indices) == 0:
        print("Nie znaleziono predykcji powyżej progu pewności. Koniec pętli.")
        break
        
    print(f"Znaleziono {len(confident_indices)} nowych pseudo-etykiet.")
    
    # Tworzenie pseudo-etykiet
    pseudo_labels = model.predict(X_unlabeled[confident_indices])
    
    # Dodawanie nowych danych do zbioru treningowego
    X_newly_labeled = X_unlabeled[confident_indices]
    
    X_labeled = np.vstack([X_labeled.toarray(), X_newly_labeled.toarray()])
    y_labeled = np.concatenate([y_labeled, pseudo_labels])
    
    # Aktualizacja puli danych nieoznaczonych (usunięcie tych, które właśnie oznaczyliśmy)
    X_unlabeled = np.delete(X_unlabeled.toarray(), confident_indices, axis=0)
    df_unlabeled = df_unlabeled.drop(df_unlabeled.index[confident_indices]).reset_index(drop=True)

print(f"\n===== ZAKOŃCZONO PROCES =====")
print(f"Finalny rozmiar zbioru treningowego: {len(y_labeled)}")

# Strategia i Najlepsze Praktyki (Kluczowe do Sukcesu!)
# Wybór Progu Pewności: To najważniejsza decyzja.
# Zacznij wysoko: Lepiej zacząć z progiem 0.99 i dodać mniej, ale bardzo pewnych etykiet, niż z progiem 0.7 i dodać dużo szumu.
# Dostosuj próg do klasy: Twój model będzie naturalnie pewniejszy co do klas z dużą liczbą przykładów (np. powder z 200 zdaniami). Możesz rozważyć dynamiczny próg – wyższy dla klas większościowych, nieco niższy dla mniejszościowych, aby dać im szansę.
# Jakość Modelu ma Znaczenie: Im lepszy Twój początkowy model, tym wyższej jakości będą pseudo-etykiety i tym skuteczniejszy będzie cały proces. Warto zainwestować w dobry model bazowy (np. fine-tuning modelu transformerowego) przed rozpoczęciem pseudo-etykietowania.
# Uważaj na Imbalans Klas: Pseudo-etykietowanie ma tendencję do wzmacniania istniejącego imbalansu. Model najczęściej będzie bardzo pewny co do klas, które już dobrze zna.
# Rozwiązanie: Po każdej iteracji możesz zastosować techniki balansowania danych, np. oversampling rzadkich klas lub undersampling częstych klas, zanim rozpoczniesz kolejne trenowanie.
# Stopniowe Dodawanie Danych: Zamiast dodawać wszystkie znalezione pseudo-etykiety naraz, czasem lepiej jest dodawać je partiami (np. 10% najbardziej pewnych w każdej iteracji). Daje to modelowi czas na "trawienie" nowej wiedzy i zapobiega gwałtownym zmianom w jego zachowaniu.

# Dla Twojego projektu, samouczenie to potężne narzędzie, które należy zastosować po etapie słabej superwizji. Pozwoli Ci to w pełni wykorzystać potencjał całego 90-tysięcznego zbioru danych, znacznie zwiększając ilość informacji dostępnych dla Twojego finalnego modelu.

