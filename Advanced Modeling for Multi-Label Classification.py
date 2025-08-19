# Jasne. Po przygotowaniu i powiększeniu zbioru danych nadszedł czas, aby wybrać odpowiednią architekturę modelu. W przypadku Twojego problemu — 200 podobnych, wielokrotnych etykiet i dużego imbalansu — standardowy klasyfikator tekstowy to za mało. Musimy sięgnąć po zaawansowane modele, które zostały zaprojektowane do radzenia sobie z takimi wyzwaniami.
# Oto przegląd najlepszych, zaawansowanych podejść modelowych, od fundamentalnych po najbardziej wyspecjalizowane.
# 1. Fundament: Modele Transformerowe (np. BERT) Adaptowane do Klasyfikacji Wieloetykietowej
# To jest absolutny punkt wyjścia i obecny standard w NLP. Zamiast używać klasycznych metod jak TF-IDF z Regresją Logistyczną, zaczynamy od modelu, który rozumie kontekst zdań. To kluczowe, aby odróżnić "puder w płynie" od "pudru w kamieniu".
# Jak to Działa?
# Standardowy model BERT jest wstępnie trenowany na ogromnej ilości tekstu, dzięki czemu rozumie gramatykę, semantykę i niuanse języka. Następnie "dostrajamy" (fine-tuning) go do naszego konkretnego zadania.
# Kluczowa Modyfikacja dla Problemu Wieloetykietowego:
# Standardowy klasyfikator na końcu ma warstwę softmax, która wymusza wybór tylko jednej z 200 klas.
# Musimy ją zastąpić warstwą z 200 neuronami, z których każdy ma funkcję aktywacji sigmoid.
# Każdy neuron sigmoid działa jak niezależny przełącznik binarny (0 lub 1), decydując "tak" lub "nie" dla każdej etykiety. To pozwala modelowi przypisać 0, 1, 2, lub więcej etykiet do jednego zdania.
# Funkcja straty również się zmienia z CategoricalCrossentropy na BinaryCrossentropy.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Załaduj model z głowicą klasyfikacyjną
model_name = "bert-base-multilingual-cased" # Dobry model wielojęzyczny
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=200, # Liczba Twoich etykiet
    problem_type="multi_label_classification" # To automatycznie konfiguruje sigmoidy!
)

# ... (dalej kod do przygotowania danych i pętli treningowej) ...

# Podczas treningu, etykiety muszą być w formacie one-hot, np.:
# zdanie "pomadka i tusz" -> [0, 1, 1, 0, ..., 0] (gdzie 1 jest na pozycjach 'lipstick' i 'mascara')

# Zalety (+)	Wady (-)
# Najwyższa Wydajność (State-of-the-Art): Rozumienie kontekstu jest kluczowe dla Twoich podobnych etykiet.	Zasobożerność: Trenowanie wymaga GPU i jest wolniejsze niż proste modele.
# Uczenie Transferowe: Model korzysta z wiedzy zdobytej na miliardach zdań, co jest bezcenne przy małej ilości danych dla rzadkich klas.	Wymaga Dużo Danych do Dostrojenia: Mimo wszystko, aby dobrze się dostroić, potrzebuje sporego zbioru danych (dlatego słaba superwizja była tak ważna).
# Elastyczność: Można go łatwo adaptować do wielu zadań NLP.	"Czarna Skrzynka": Trudniej jest zinterpretować, dlaczego model podjął daną decyzję.
# 2. Specjalista od Struktury: Klasyfikacja Hierarchiczna
# To podejście jest idealne, jeśli Twoje 200 etykiet da się zorganizować w strukturę drzewa (hierarchię).

# - Makijaż
#   - Twarz
#     - Podkład
#       - Podkład w płynie
#       - Podkład w sztyfcie
#     - Puder
#   - Usta
#     - Pomadka
#     - Błyszczyk
# - Pielęgnacja
#   - Nawilżanie
#     - Krem
#     - Serum

# Jak to Działa?
# Zamiast jednego potężnego modelu, który wybiera jedną z 200 etykiet, budujesz serię mniejszych, prostszych klasyfikatorów:
# Klasyfikator Poziomu 1: Decyduje, czy zdanie dotyczy "Makijażu" czy "Pielęgnacji".
# Klasyfikator Poziomu 2: Jeśli Poziom 1 wybrał "Makijaż", ten klasyfikator decyduje, czy chodzi o "Twarz" czy "Usta".
# Klasyfikator Poziomu 3: I tak dalej, aż do liścia drzewa (konkretnej etykiety).
# Zalety (+)	Wady (-)
# Rozbicie Problemu: Zamiast jednego mega-trudnego zadania, model rozwiązuje serię łatwiejszych. To często prowadzi do wyższej dokładności.	Wymaga Zdefiniowania Hierarchii: Musisz ręcznie stworzyć i utrzymywać strukturę drzewa etykiet.
# Lepsze Radzenie Sobie z Imbalansem: Klasyfikatory na wyższych poziomach drzewa widzą znacznie więcej przykładów, co stabilizuje ich trening.	Propagacja Błędu: Błąd na wysokim poziomie hierarchii (np. błędna klasyfikacja "Makijażu" jako "Pielęgnacji") jest nie do odzyskania na niższych poziomach.
# Skalowalność: Łatwiej jest dodać nową etykietę (np. nowy rodzaj podkładu) bez konieczności trenowania całego systemu od zera.	Złożoność Implementacji: Wymaga zarządzania wieloma modelami i logiką przepływu danych między nimi.
# 3. Specjalista od Niunansów: Sieci z Uwagą Zależną od Etykiety (LWANs)
# To jest bardzo zaawansowane i eleganckie rozwiązanie, idealnie pasujące do Twojego problemu z wieloma podobnymi etykietami.
# Intuicja:
# Standardowy mechanizm uwagi (attention) w BERT decyduje, które słowa w zdaniu są ogólnie ważne. Ale w Twoim przypadku, ważność słowa zależy od etykiety, o którą pytamy!
# W zdaniu: "Ta wodoodporna formuła wydłuża rzęsy, a jej matowe wykończenie na ustach jest piękne."
# Dla etykiety mascara, kluczowe są słowa "wodoodporna", "wydłuża", "rzęsy".
# Dla etykiety lipstick, kluczowe są słowa "matowe", "ustach".
# Jak to Działa?
# LWAN (Label-Wise Attention Network) tworzy osobny mechanizm uwagi dla każdej etykiety. Model uczy się, na które fragmenty tekstu patrzeć, aby podjąć decyzję dla lipstick, a na które, aby podjąć decyzję dla mascara. Zazwyczaj jest to warstwa uwagi dodana na wyjściu z modelu bazowego (np. BERT).
# Zalety (+)	Wady (-)
# Idealne do Klasyfikacji Drobnoziarnistej: Zaprojektowane do odróżniania bardzo podobnych klas.	Złożoność Obliczeniowa: Trenowanie osobnego mechanizmu uwagi dla każdej z 200 etykiet jest kosztowne.
# Wysoka Interpretowalność: Możesz zwizualizować, które słowa w zdaniu "aktywowały" daną etykietę, co jest bezcenne przy analizie błędów.	Ryzyko Przeuczenia: Przy małej ilości danych dla rzadkich klas, dedykowany mechanizm uwagi może nauczyć się na pamięć szumu zamiast prawdziwych wzorców.
# Technika Treningowa do Zastosowania z Każdym Modelem: Focal Loss
# To nie jest model, ale funkcja straty, która jest niezwykle skuteczna w walce z imbalansem klas. Jest to "tuning" dla BinaryCrossentropy.
# Jak to Działa?
# Standardowa funkcja straty traktuje wszystkie błędy jednakowo. Focal Loss modyfikuje ją tak, aby zmniejszyć wagę straty dla przykładów, które są łatwe do sklasyfikowania (np. model jest już w 99% pewny, że zdanie jest o pomadce). To zmusza model do skupienia całej swojej "uwagi" na nauce trudnych przypadków i przykładów z klas mniejszościowych. Jest to idealne rozwiązanie dla Twojego problemu z 10% recallu dla klasy "relevant" i rzadkich etykiet.

# Rekomendacja i Podsumowanie
# Podejście	Najlepsze Dla	Złożoność
# BERT z Głowicą Multi-Label	Punkt startowy i solidna podstawa. Najlepszy ogólny wybór, który zapewni bardzo dobre wyniki.	Średnia
# ... z użyciem Focal Loss	Rozwiązanie problemu imbalansu. Niezbędny dodatek do powyższego, aby podbić wyniki na rzadkich klasach.	Średnia (dodanie innej funkcji straty)
# Klasyfikacja Hierarchiczna	Sytuacji, gdy 200 etykiet ma jasną, naturalną strukturę drzewa. Może pobić BERT, jeśli hierarchia jest dobra.	Wysoka
# LWAN (Label-Wise Attention)	Ostatecznego "wyciśnięcia" dokładności, gdy kluczowe jest odróżnianie bardzo subtelnych różnic między etykietami.	Bardzo Wysoka

Rekomendowana strategia:
# Zacznij od modelu BERT adaptowanego do klasyfikacji wieloetykietowej. To Twój solidny fundament.
# Od razu zaimplementuj trening z użyciem Focal Loss zamiast standardowej BinaryCrossentropy. To bezpośrednio zaatakuje Twój największy problem – imbalans klas i niski recall.
# Jeśli po osiągnięciu dobrych wyników z powyższym podejściem nadal będziesz potrzebować wyższej precyzji, rozważ zaimplementowanie LWAN jako dodatkowej warstwy na szczycie BERTa, aby jeszcze lepiej radzić sobie z niuansami między Twoimi 200 etykietami.
