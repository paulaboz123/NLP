# Oczywiście. Przechodzimy teraz do strategii, która jest bardziej zmianą w architekturze całego systemu niż pojedynczą techniką modelowania. Mowa o Osobnym Klasyfikatorze Binarnym (Separate Binary Classifier), często nazywanym modelem dwuetapowym (two-stage model) lub kaskadowym (cascade model).
# Ta strategia bezpośrednio atakuje jeden z Twoich największych problemów: bardzo niski recall (10%) dla klasy "relevant". Oznacza to, że Twój obecny model przepuszcza 90% zdań, które powinien wychwycić.
# Główna Idea: Rozbij Duży Problem na Dwa Mniejsze
# Zamiast budować jeden, mega-skomplikowany model, który musi jednocześnie robić dwie rzeczy:
# Odróżnić zdania relewantne od tysięcy nierelewantnych.
# Przypisać jedną z 200 bardzo podobnych etykiet do tych relewantnych.
# ...rozbijasz ten proces na dwa prostsze, wyspecjalizowane etapy:
# Etap 1: "Bramkarz" - Klasyfikator Binarny Relewantności
# Zadanie: Ten model ma tylko jedno, proste zadanie: odpowiedzieć na pytanie "Czy to zdanie jest w ogóle o kosmetykach/pielęgnacji, czy to kompletny śmieć?".
# Klasy: Dwie klasy: relevant (1) i not_relevant (0).
# Dane Treningowe: Wszystkie Twoje 3000 oznaczonych zdań mają etykietę relevant. Wszystkie pozostałe 87 000 zdań można na początku potraktować jako not_relevant.
# Etap 2: "Specjalista" - Klasyfikator Wieloetykietowy
# Zadanie: Ten model otrzymuje na wejściu tylko te zdania, które "Bramkarz" wpuścił (oznaczył jako relevant). Jego zadaniem jest przypisanie do nich jednej lub więcej z 200 szczegółowych etykiet.
# Klasy: 200 klas (np. lipstick, powder, mascara...).
# Dane Treningowe: Tylko Twoje 3000 oznaczonych zdań, ponieważ tylko one mają szczegółowe etykiety.
# Dlaczego to jest tak skuteczne?
# Skupienie na Problemie (Focus):
# "Bramkarz" może być zoptymalizowany pod kątem jednego celu: maksymalizacji recallu dla klasy relevant. Możesz użyć technik takich jak ważenie klas, oversampling, czy dostosowanie progu decyzyjnego, aby upewnić się, że przepuszcza on jak najmniej fałszywych negatywów (zdania relewantne oznaczone jako nierelewantne). Nawet jeśli przy okazji wpuści trochę śmieci (niska precyzja), to jest to mniejszy problem.
# "Specjalista" nie musi już marnować swojej "mocy obliczeniowej" i uwagi na odróżnianie zdań o kosmetykach od zdań o samochodach. Może w pełni skupić się na trudnym zadaniu odróżniania powder od liquid powder. Działa na znacznie "czystszych", bardziej skoncentrowanych danych.
# Lepsze Zarządzanie Imbalansem:
# Imbalans w pierwszym etapie jest ogromny (3k vs 87k), ale jest to problem binarny, z którym łatwiej walczyć za pomocą standardowych technik (np. class_weight='balanced').
# W drugim etapie imbalans wciąż istnieje (200 vs 20 przykładów), ale model nie jest już "rozpraszany" przez 97% danych, które są kompletnie nie na temat.
# Wydajność Obliczeniowa:
# "Bramkarz" może być bardzo szybkim i lekkim modelem (np. Regresja Logistyczna na TF-IDF, FastText), ponieważ jego zadanie jest stosunkowo proste.
# Potężny, zasobożerny model (jak BERT) uruchamiasz tylko na małym odsetku danych, które przejdą przez pierwszy etap, co znacząco przyspiesza cały proces predykcji.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# W praktyce, model_multilabel byłby modelem transformerowym

# --- Krok 1: Przygotowanie Danych ---
# Załóżmy, że df zawiera wszystkie 90k wierszy
# df['relevance'] ma wartość 1 dla 3k zdań, 0 dla reszty
# df['label'] ma nazwy etykiet dla 3k zdań, a NaN dla reszty

# Dane do treningu "Bramkarza" (klasyfikatora binarnego)
X_binary = df['text']
y_binary = df['relevance']

# Dane do treningu "Specjalisty" (klasyfikatora wieloetykietowego)
df_multilabel_train = df.dropna(subset=['label'])
X_multilabel = df_multilabel_train['text']
y_multilabel = df_multilabel_train['label'] # To wymagałoby dalszego przetwarzania (MultiLabelBinarizer)

# --- Krok 2: Trenowanie Modeli ---

# Trenowanie "Bramkarza"
print("Trenowanie klasyfikatora binarnego ('Bramkarza')...")
vectorizer_binary = TfidfVectorizer(max_features=5000)
X_binary_vec = vectorizer_binary.fit_transform(X_binary)
# Użyj ważenia klas, aby walczyć z imbalansem!
model_binary = LogisticRegression(class_weight='balanced', max_iter=1000)
model_binary.fit(X_binary_vec, y_binary)
print("... Zakończono.")

# Trenowanie "Specjalisty" (logika koncepcyjna)
print("Trenowanie klasyfikatora wieloetykietowego ('Specjalisty')...")
# ... tutaj nastąpiłby cały proces fine-tuningu BERTa na X_multilabel i y_multilabel ...
model_multilabel = # Wytrenowany, potężny model transformerowy
print("... Zakończono.")


# --- Krok 3: Logika Predykcji Kaskadowej ---

def predict_cascade(texts_to_predict):
    final_predictions = []
    
    # Etap 1: "Bramkarz" filtruje dane
    texts_vec = vectorizer_binary.transform(texts_to_predict)
    relevance_preds = model_binary.predict(texts_vec)
    
    relevant_texts = [text for text, pred in zip(texts_to_predict, relevance_preds) if pred == 1]
    
    # Etap 2: "Specjalista" działa tylko na odfiltrowanych danych
    if relevant_texts:
        multilabel_preds = model_multilabel.predict(relevant_texts)
    else:
        multilabel_preds = []
        
    # Połącz wyniki
    pred_idx = 0
    for rel_pred in relevance_preds:
        if rel_pred == 1:
            final_predictions.append({'text': texts_to_predict[pred_idx], 'relevance': 1, 'labels': multilabel_preds[pred_idx]})
            pred_idx += 1
        else:
            final_predictions.append({'text': texts_to_predict[pred_idx], 'relevance': 0, 'labels': []})

    return final_predictions

# Użycie:
new_sentences = ["Ten nowy tusz do rzęs jest niesamowity!", "Wczoraj padał deszcz.", "Polecam ten podkład."]
results = predict_cascade(new_sentences)
# Oczekiwany wynik: dwa pierwsze zdania zostaną przepuszczone do specjalisty, a drugie odrzucone.

# Strategia i Miejsce w Twoim Projekcie
# Architektura dwuetapowa to fundamentalna decyzja projektowa, którą warto podjąć na wczesnym etapie modelowania.
# Kiedy Stosować?: Jest idealna dla Twojej sytuacji, gdzie:
# Istnieje ogromna dysproporcja między danymi relewantnymi a nierelewantnymi.
# Recall dla klasy relewantnej jest krytycznie niski.
# Zadanie klasyfikacji szczegółowej jest bardzo złożone (wiele podobnych klas).
# Strojenie "Bramkarza": Cały sukces tej metody zależy od tego, jak dobrze skonfigurujesz pierwszy etap. Musisz znaleźć złoty środek między recallem a precyzją. Zazwyczaj w takich systemach dąży się do bardzo wysokiego recallu (np. 98%), nawet kosztem niższej precyzji (np. 50%). Oznacza to, że system wychwyci prawie wszystkie relewantne zdania, ale przepuści też trochę "śmieci". To jest w porządku, ponieważ "Specjalista" w drugim etapie może mieć klasę "żadna z powyższych" lub po prostu zwrócić niskie prawdopodobieństwa dla wszystkich 200 klas.
# To podejście jest jedną z najbardziej niezawodnych i sprawdzonych metod na radzenie sobie z problemem "szukania igły w stogu siana", który jest sercem Twojego zadania. Zamiast budować jeden monolit, budujesz linię montażową, w której każdy element jest ekspertem w swojej wąskiej dziedzinie.
