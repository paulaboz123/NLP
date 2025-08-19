# Oczywiście. Przechodzimy do Uczenia Wrażliwego na Koszt (Cost-Sensitive Learning). Jest to zbiór technik, które pozwalają bezpośrednio "powiedzieć" modelowi podczas treningu, że niektóre błędy są znacznie gorsze niż inne.
# W Twoim przypadku, błąd polegający na tym, że model oznacza relewantne zdanie jako nierelewantne (Fałszywy Negatyw - False Negative) jest znacznie bardziej kosztowny niż błąd, w którym oznacza nierelewantne zdanie jako relewantne (Fałszywy Pozytyw - False Positive). Dlaczego? Bo w pierwszym przypadku tracisz informację na zawsze, a w drugim co najwyżej zaprezentujesz użytkownikowi niepotrzebne podświetlenie.
# Cost-Sensitive Learning to bezpośredni i elegancki sposób na rozwiązanie tego problemu, zwłaszcza niskiego recallu dla klasy relevant.
# Główna Idea: Wprowadź Kary Finansowe do Treningu
# Wyobraź sobie, że trenujesz pracownika, a on popełnia dwa rodzaje błędów:
# Błąd typu A: Zapomina wysłać klientowi faktury na 10 zł. (Mały koszt)
# Błąd typu B: Zapomina wysłać klientowi faktury na 10 000 zł. (Duży koszt)
# Standardowy trening modelu jest jak mówienie pracownikowi: "Każdy błąd jest tak samo zły". W rezultacie pracownik może zoptymalizować swoją pracę, aby unikać małych, częstych błędów, ignorując te rzadkie, ale katastrofalne w skutkach.
# Cost-Sensitive Learning to jak wprowadzenie systemu kar: "Za błąd typu A potrącę Ci 10 zł z pensji. Za błąd typu B potrącę Ci 10 000 zł". Pracownik (model) bardzo szybko nauczy się, że musi szczególnie uważać, aby nie popełnić błędu typu B.
# W uczeniu maszynowym robimy to, modyfikując funkcję straty (loss function).
# Jak to Działa w Praktyce? Ważenie Klas (Class Weighting)
# Najprostszą i najpopularniejszą metodą implementacji uczenia wrażliwego na koszt jest ważenie klas.
# Jak działa?
# Podczas obliczania błędu (straty) na koniec każdej partii (batch) danych, błędy popełnione na przykładach z klas mniejszościowych lub ważniejszych są mnożone przez większą wagę.
# Standardowa strata: Loss_total = Loss(przykład_1) + Loss(przykład_2) + ...
# Strata z wagami: Loss_total = waga_1 * Loss(przykład_1) + waga_2 * Loss(przykład_2) + ...
# Jeśli przykład_1 należy do rzadkiej, ale ważnej klasy relevant, jego waga_1 będzie znacznie większa (np. 10.0), a waga_2 dla częstej klasy not_relevant będzie mniejsza (np. 0.8).
# Efekt: Model otrzymuje znacznie większą "karę" za pomyłkę na przykładzie z klasy relevant. Aby zminimalizować całkowitą stratę, będzie musiał poświęcić znacznie więcej uwagi na poprawne klasyfikowanie tej klasy. To w naturalny sposób prowadzi do zwiększenia recallu dla klasy relevant.
# Zastosowanie w Twoim Projekcie
# Masz dwa główne problemy z imbalansem, gdzie ta technika jest idealna:
# 1. Poprawa Recallu dla Klasy relevant (w "Bramkarzu")
# Problem: Masz 3k przykładów relevant (klasa mniejszościowa) vs 87k not_relevant (klasa większościowa).
# Rozwiązanie: Podczas trenowania swojego binarnego klasyfikatora relewantności, ustaw wagi klas. Wiele bibliotek, w tym scikit-learn i Hugging Face Transformers, pozwala to zrobić jednym parametrem.
# Jak obliczyć wagi?: Najprostsza metoda to odwrotność proporcji klas.
# Liczba próbek: N = 90000
# Liczba klas: k = 2
# waga_klasy_j = N / (k * liczba_próbek_w_klasie_j)
# waga_relevant = 90000 / (2 * 3000) = 15.0
# waga_not_relevant = 90000 / (2 * 87000) ≈ 0.52
# Model będzie "karany" prawie 30 razy mocniej za błąd na przykładzie relewantnym!
# 2. Poprawa Klasyfikacji Rzadkich Etykiet (w "Specjaliście")
# Problem: Masz 200 przykładów powder vs 20 liquid powder.
# Rozwiązanie: Podczas trenowania swojego wieloetykietowego klasyfikatora (np. BERTa), zamiast standardowej BinaryCrossentropy, użyj jej ważonej wersji (WeightedBinaryCrossentropy). Obliczasz wagi dla każdej z 200 klas i przekazujesz je do funkcji straty.
# Efekt: Model nauczy się, że pomyłka na liquid powder jest znacznie "droższa" niż pomyłka na powder, co zmusi go do zwracania większej uwagi na cechy wyróżniające tę rzadką klasę.
# Kod Koncepcyjny
# W scikit-learn (dla "Bramkarza")
# To jest trywialnie proste. Wystarczy jeden parametr w konstruktorze modelu.


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Większość klasyfikatorów w scikit-learn ma ten parametr
# 'balanced' automatycznie oblicza wagi na podstawie odwrotności proporcji
model = LogisticRegression(class_weight='balanced', max_iter=1000)

# Można też podać wagi ręcznie jako słownik
# model = SVC(class_weight={1: 15.0, 0: 0.52}) # Gdzie 1 to 'relevant', a 0 to 'not_relevant'

# ... (reszta kodu do trenowania) ...

# W PyTorch/Transformers (dla "Specjalisty")
# Wymaga to nieco więcej pracy, ponieważ musisz przekazać wagi bezpośrednio do funkcji straty.
import torch
import torch.nn as nn

# Załóżmy, że mamy niezbalansowane wyjścia i etykiety
# outputs.shape: (batch_size, num_classes) -> (4, 200)
# labels.shape: (batch_size, num_classes) -> (4, 200)

# 1. Oblicz wagi dla każdej z 200 klas (robisz to raz, przed treningiem)
# class_weights to tensor o długości 200, np. [0.5, 15.0, 8.2, ...]
class_weights = calculate_class_weights_for_200_labels() 

# 2. Zdefiniuj funkcję straty
# `pos_weight` to parametr w BCEWithLogitsLoss, który obsługuje ważenie
# Oczekuje on tensora o tej samej długości co liczba klas
loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weights)

# 3. Użyj w pętli treningowej
# loss = loss_function(outputs, labels)
# loss.backward()
# ...
#  Hugging Face Trainer można to osiągnąć, tworząc podklasę Trainer i nadpisując jej metodę compute_loss.
# Podsumowanie
# Zalety (+)	Wady (-)
# Bezpośrednia Kontrola nad Kosztem Błędów: Mówisz modelowi wprost, na czym ma się skupić. To bardziej bezpośrednie niż augmentacja czy oversampling.	Trudność w Doborze Wag: Znalezienie optymalnych wag może wymagać eksperymentów. Automatyczne obliczanie to dobry start, ale ręczne dostrojenie może dać lepsze wyniki.
# Efektywność Obliczeniowa: Nie tworzy nowych danych (jak augmentacja) ani nie usuwa istniejących (jak undersampling). Działa na oryginalnym zbiorze, co jest wydajne.	Może Obniżyć Ogólną Dokładność: Zmuszając model do skupienia się na rzadkiej klasie, możesz nieco pogorszyć jego wydajność na klasie większościowej. Jest to trade-off (coś za coś) między recalem a precyzją.
# Łatwość Implementacji: W wielu bibliotekach sprowadza się to do ustawienia jednego parametru.	Nie Tworzy Nowej Informacji: W przeciwieństwie do augmentacji, ta metoda nie pokazuje modelowi nowych wariacji danych. Po prostu każe mu bardziej "przyglądać się" tym, które już ma.
# Miejsce w Twoim Projekcie: Cost-Sensitive Learning to technika treningowa, którą stosujesz w momencie trenowania swoich modeli. Jest to jedna z pierwszych i najważniejszych metod, które powinieneś zastosować, aby walczyć z imbalansem, zaraz obok (lub w połączeniu z) Focused Data Augmentation. Jest szczególnie potężna w architekturze dwuetapowej do "podkręcenia" recallu Twojego "Bramkarza".

