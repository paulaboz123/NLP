# Oczywiście. Przechodzimy do dwóch bardzo zaawansowanych architektur modelowania, które są specjalnie zaprojektowane do rozwiązywania problemów takich jak Twój: Klasyfikacja Hierarchiczna i Sieci z Uwagą Zależną od Etykiety (LWANs). Obie metody wykraczają poza standardowe podejście "płaskiej" klasyfikacji i próbują w inteligentny sposób zarządzać dużą liczbą etykiet.
# 1. Klasyfikacja Hierarchiczna (Hierarchical Classification)
# Ta metoda jest skuteczna tylko wtedy, gdy Twoje 200 etykiet można sensownie zorganizować w strukturę drzewiastą lub graf acykliczny (DAG). Musisz najpierw zainwestować czas w stworzenie tej hierarchii.
# Krok 1: Zdefiniuj Hierarchię
# Załóżmy, że po analizie Twoich etykiet, tworzysz taką strukturę:


# (ROOT)
# ├── Makijaż
# │   ├── Twarz
# │   │   ├── Podkład
# │   │   ├── Puder
# │   │   └── Korektor
# │   └── Oczy
# │       ├── Tusz do rzęs
# │       └── Cień do powiek
# └── Pielęgnacja
#     ├── Twarz
#     │   ├── Krem nawilżający
#     │   └── Serum
#     └── Ciało
#         └── Balsam

# Krok 2: Wybierz Architekturę Modelu
# Istnieją dwa główne podejścia:
# A) Podejście Lokalne: Seria Niezależnych Klasyfikatorów (Local Classifiers per Parent Node)
# To jest najbardziej intuicyjne i najczęstsze podejście.
# Jak to działa?: Trenujesz osobny, mały klasyfikator dla każdego węzła w drzewie, który ma dzieci.
# Klasyfikator ROOT: Bada zdanie i decyduje: "Makijaż" czy "Pielęgnacja"? (Klasyfikacja binarna).
# Klasyfikator Makijaż: Uruchamia się tylko, jeśli ROOT wybrał "Makijaż". Decyduje: "Twarz" czy "Oczy"? (Klasyfikacja binarna).
# Klasyfikator Twarz (w Makijażu): Uruchamia się, jeśli Makijaż wybrał "Twarz". Decyduje: "Podkład", "Puder" czy "Korektor"? (Klasyfikacja wieloklasowa).
# Predykcja: Aby sklasyfikować nowe zdanie, przechodzisz od korzenia w dół, używając predykcji na każdym poziomie, aby zdecydować, który klasyfikator uruchomić na następnym poziomie.
# Kod Koncepcyjny (logika, nie pełny kod treningowy):


# Załóżmy, że mamy wytrenowane modele i zapisane w słowniku
# klucz to węzeł-rodzic, wartość to wytrenowany model
trained_classifiers = {
    'ROOT': model_root,
    'Makijaż': model_makijaz,
    'Pielęgnacja': model_pielegnacja,
    'Makijaż/Twarz': model_makijaz_twarz,
    # ... i tak dalej
}

def predict_hierarchically(text, hierarchy_tree):
    path = ['ROOT']
    current_node = 'ROOT'
    
    while current_node in trained_classifiers:
        # Wybierz odpowiedni, wstępnie wytrenowany klasyfikator
        classifier = trained_classifiers[current_node]
        
        # Przewiduj następny krok w hierarchii
        prediction = classifier.predict(text) # np. 'Makijaż'
        
        # Dodaj do ścieżki i przejdź głębiej
        path.append(prediction)
        current_node = '/'.join(path[1:]) # np. 'Makijaż/Twarz'
        
        # Jeśli dotarliśmy do liścia (końcowej etykiety), zakończ
        if current_node not in trained_classifiers:
            return path[1:] # Zwróć całą ścieżkę jako etykiety
            
    return path[1:]

# B) Podejście Globalne: Jeden Model dla Całej Hierarchii (Global Classifier)
# Jak to działa?: Używasz jednego, potężnego modelu (np. BERT), który ma na końcu zmodyfikowaną warstwę wyjściową. Model jest trenowany, aby przewidywać całą ścieżkę od korzenia do liścia za jednym razem. Funkcja straty jest tak skonstruowana, że kara za błąd jest większa, jeśli pomyłka nastąpiła wysoko w drzewie (np. pomylenie "Makijażu" z "Pielęgnacją" jest gorsze niż pomylenie "Podkładu" z "Pudrem").
# Złożoność: Znacznie trudniejsze w implementacji. Wymaga niestandardowej funkcji straty, która rozumie strukturę hierarchii.
# Rekomendacja: Zacznij od podejścia lokalnego (A). Jest ono znacznie prostsze w implementacji, debugowaniu i pozwala na użycie standardowych modeli klasyfikacyjnych na każdym etapie.
# 2. Sieci z Uwagą Zależną od Etykiety (LWANs - Label-Wise Attention Networks)
# Ta architektura nie wymaga predefiniowanej hierarchii. Zamiast tego, uczy się, że różne etykiety zależą od różnych słów w tym samym zdaniu.
# Architektura Krok po Kroku
# Warstwa Enkodera Zdań (Sentence Encoder):
# To jest Twój standardowy "silnik" do rozumienia tekstu. Najczęściej jest to model oparty na architekturze Bi-LSTM lub, co jest znacznie lepsze, wstępnie wytrenowany BERT.
# Wejście: Surowe zdanie.
# Wyjście: Sekwencja wektorów (osadzeń), po jednym dla każdego słowa w zdaniu. Te wektory są "kontekstowe" – osadzenie słowa "puder" jest inne w zależności od reszty zdania. Nazwijmy te wyjścia H.
# Warstwa Uwagi Zależnej od Etykiety (Label-Wise Attention Layer):
# To jest serce tej architektury i kluczowa różnica.
# Inicjalizacja: Na początku tworzysz wektor osadzenia dla każdej z 200 etykiet. Te wektory są parametrami modelu i będą się uczyć podczas treningu. Nazwijmy je L_1, L_2, ..., L_200.
# Obliczenia: Dla każdej etykiety L_i z osobna, model wykonuje następujące operacje:
# a. Oblicza "dopasowanie" między wektorem etykiety L_i a każdym wektorem słowa h_j z wyjścia enkodera H.
# b. Na podstawie tych dopasowań, oblicza wagi uwagi alpha_i – zestaw liczb mówiących, jak ważne jest każde słowo w zdaniu dla tej konkretnej etykiety L_i.
# c. Tworzy jeden wektor v_i, który jest ważoną sumą wektorów słów H, używając wag alpha_i. Ten wektor v_i to reprezentacja zdania "z punktu widzenia" etykiety i.
# Warstwa Wyjściowa:
# Masz teraz 200 różnych wektorów v_1, v_2, ..., v_200 – po jednym dla każdej etykiety.
# Każdy wektor v_i jest przepuszczany przez prosty, niezależny klasyfikator binarny (zazwyczaj pojedyncza warstwa z aktywacją sigmoid), który decyduje "tak" lub "nie" dla etykiety i.
# Kod Koncepcyjny (z użyciem PyTorch, mocno uproszczony):

import torch
import torch.nn as nn
import torch.nn.functional as F

class LWAN(nn.Module):
    def __init__(self, sentence_encoder, num_labels, hidden_dim):
        super(LWAN, self).__init__()
        self.encoder = sentence_encoder
        self.num_labels = num_labels
        
        # Uczące się osadzenia dla każdej etykiety
        self.label_embeddings = nn.Embedding(num_labels, hidden_dim)
        
        # Warstwa do obliczania dopasowania (uwagi)
        self.attention_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Finalne klasyfikatory binarne dla każdej etykiety
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_labels)])

    def forward(self, text_input):
        # 1. Enkoder zdania
        # H ma wymiar (batch_size, seq_length, hidden_dim)
        H = self.encoder(text_input).last_hidden_state 

        outputs = []
        for i in range(self.num_labels):
            # 2. Warstwa uwagi
            # Pobierz osadzenie dla i-tej etykiety
            label_emb = self.label_embeddings(torch.tensor(i)) # Wymiar: (hidden_dim)
            
            # Oblicz dopasowanie między etykietą a każdym słowem
            # Uproszczone obliczenie uwagi
            attention_scores = torch.tanh(self.attention_layer(H)) # (batch, seq, hidden)
            attention_scores = torch.matmul(attention_scores, label_emb.unsqueeze(1)) # (batch, seq, 1)
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Stwórz wektor zdania "z punktu widzenia" i-tej etykiety
            sentence_vector_for_label_i = torch.sum(H * attention_weights, dim=1) # (batch, hidden)
            
            # 3. Warstwa wyjściowa
            # Użyj i-tego klasyfikatora
            output = self.output_layers[i](sentence_vector_for_label_i)
            outputs.append(output)
            
        # Złącz wyniki i przepuść przez sigmoid
        final_output = torch.cat(outputs, dim=1)
        return torch.sigmoid(final_output)

# Użycie:
# bert_model = AutoModel.from_pretrained(...)
# lwan_model = LWAN(sentence_encoder=bert_model, num_labels=200, hidden_dim=768)

# Podsumowanie i Strategia dla Ciebie
# Metoda	Kiedy Używać?	Zalety	Wady	Złożoność Implementacji
# Klasyfikacja Hierarchiczna	Gdy Twoje 200 etykiet ma jasną, naturalną, drzewiastą strukturę.	Dzieli problem na mniejsze, potencjalnie poprawiając dokładność i radzenie sobie z imbalansem.	Wymaga ręcznego stworzenia hierarchii; błędy na wyższych poziomach propagują się w dół.	Średnia do Wysokiej (zwłaszcza w wersji globalnej)
# LWANs	Gdy masz wiele podobnych, drobnoziarnistych etykiet i kluczowe jest znalezienie subtelnych różnic w tekście.	Uczy się, które słowa są ważne dla której etykiety; wysoka interpretowalność; nie wymaga hierarchii.	Kosztowna obliczeniowo; może być trudna w trenowaniu i podatna na przeuczenie przy małej ilości danych.	Wysoka
# Rekomendowana Strategia:

# Zacznij od analizy etykiet: Czy możesz bez większych problemów narysować drzewo swoich 200 etykiet? Jeśli odpowiedź brzmi "tak" i ta struktura jest dla Ciebie logiczna, Klasyfikacja Hierarchiczna (w podejściu lokalnym) jest bardzo obiecującym kierunkiem. Może znacząco uprościć Twój problem.
# Jeśli hierarchia jest niejasna lub sztuczna: Jeśli etykiety tworzą bardziej "płaską" lub skomplikowaną sieć powiązań, nie zmuszaj się do tworzenia hierarchii. W takim przypadku LWAN jest architekturą stworzoną dla Ciebie. Zaimplementowanie jej na szczycie modelu BERT to potężne podejście do radzenia sobie z niuansami w Twoich danych.
# Pragmatyczny start: Zawsze możesz zacząć od najprostszego, ale wciąż mocnego modelu (BERT z głowicą wieloetykietową), a dopiero potem, w celu poprawy wyników, przejść do implementacji jednej z tych bardziej złożonych architektur.
