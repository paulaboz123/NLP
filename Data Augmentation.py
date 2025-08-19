Oczywiście. Przechodzimy do Augmentacji Danych (Data Augmentation). Jest to zbiór technik, których celem jest sztuczne powiększenie zbioru treningowego poprzez tworzenie nowych, zmodyfikowanych kopii istniejących danych.
W kontekście Twojego projektu, augmentacja jest absolutnie kluczowa, aby rozwiązać jeden z Twoich największych problemów: dramatyczny imbalans klas, gdzie niektóre etykiety mają 200 przykładów, a inne zaledwie 20.
Główna Idea: Pokaż Modelowi Różne Sposoby Mówienia Tego Samego
Wyobraź sobie, że uczysz dziecko rozpoznawać "psa". Jeśli pokażesz mu 20 identycznych zdjęć tego samego golden retrievera, dziecko nauczy się rozpoznawać tylko tego konkretnego psa.
Zamiast tego, pokazujesz mu:
Zdjęcie z innej perspektywy.
Zdjęcie czarno-białe.
Zdjęcie lekko obrócone.
Zdjęcie z innym tłem.
To są augmentacje. Dziecko (model) uczy się, że te wszystkie wariacje wciąż oznaczają "psa". W rezultacie uczy się konceptu psa, a nie tylko jednego, konkretnego obrazu. Staje się bardziej odporne (robust) na nowe, niewidziane wcześniej przykłady.
W NLP robimy dokładnie to samo, ale operujemy na tekście. Chcemy nauczyć model, że "Ten tusz do rzęs jest fantastyczny" i "Uważam, że ta maskara jest wspaniała" znaczą to samo, mimo że używają innych słów.
Dlaczego to jest tak ważne dla Twojego projektu?
Walka z Imbalansem Klas: To jest główne zastosowanie. Dla etykiety liquid powder, która ma tylko 20 przykładów, możesz wygenerować 5-10 wariacji dla każdego zdania, sztucznie zwiększając liczbę przykładów do 100-200. To daje modelowi znacznie więcej materiału do nauki dla tej rzadkiej klasy.
Zwiększenie Odporności Modelu (Robustness): Model, który widział wiele różnych sformułowań tego samego konceptu, będzie mniej wrażliwy na drobne zmiany w tekście (synonimy, literówki, inny szyk zdania) i będzie lepiej generalizował na nowe dane.
Redukcja Przeuczenia (Overfitting): Szczególnie w przypadku rzadkich klas, model ma tendencję do "uczenia się na pamięć" tych kilku przykładów, które widział. Augmentacja wprowadza różnorodność, która zmusza model do nauki bardziej ogólnych wzorców.
Najpopularniejsze Techniki Augmentacji Tekstu
Oto przegląd metod, od najprostszych do najbardziej zaawansowanych.
1. EDA: Easy Data Augmentation
To zbiór czterech prostych, ale często skutecznych operacji, które nie wymagają zaawansowanych modeli.
Synonym Replacement (SR): Zastąp losowe słowa (ale nie "stop-words") ich synonimami.
Oryginał: "Ten podkład daje piękne, matowe wykończenie."
Augmentacja: "Ten podkład daje wspaniałe, niebłyszczące wykończenie."
Random Insertion (RI): Wstaw losowo synonimy innych słów w zdaniu.
Oryginał: "Ten podkład daje piękne, matowe wykończenie."
Augmentacja: "Ten podkład naprawdę daje piękne, matowe dobre wykończenie."
Random Swap (RS): Zamień miejscami dwa losowe słowa w zdaniu.
Oryginał: "Ten podkład daje piękne, matowe wykończenie."
Augmentacja: "Ten wykończenie daje piękne, matowe podkład." (Ryzykowne, może zepsuć gramatykę).
Random Deletion (RD): Usuń losowe słowa ze zdania.
Oryginał: "Ten podkład daje piękne, matowe wykończenie."
Augmentacja: "Ten podkład piękne, matowe wykończenie."
2. Back Translation (Tłumaczenie w Obie Strony)
Bardzo popularna i skuteczna technika.
Jak działa?: Tłumaczysz zdanie z języka oryginalnego (np. polskiego) na język pośredni (np. angielski, niemiecki, japoński), a następnie tłumaczysz je z powrotem na język oryginalny.
Wynik: Otrzymujesz zdanie, które ma to samo znaczenie, ale często używa innych synonimów i ma inną strukturę składniową.
Oryginał: "Ten tusz do rzęs jest absolutnie najlepszy, jakiego kiedykolwiek używałam."
PL -> EN: "This mascara is a absolutely the best I have ever used."
EN -> PL: "Ta maskara jest absolutnie najlepsza, jakiej kiedykolwiek używałam." (W tym przypadku podobne, ale często wychodzą ciekawe parafrazy).
3. Augmentacja z Użyciem Modeli Językowych (np. BERT, GPT)
To najnowocześniejsze i najbardziej elastyczne podejście.
Masked Language Model (np. BERT):
Jak działa?: Bierzesz zdanie, losowo "maskujesz" niektóre słowa (zastępujesz je tokenem [MASK]), a następnie prosisz model BERT, aby przewidział, jakie słowa najlepiej pasują w te puste miejsca.
Oryginał: "Ten podkład daje [MASK], matowe [MASK]."
Predykcje BERTa: Może zasugerować ["piękne", "wykończenie"], ["wspaniałe", "efekt"], ["dobre", "pokrycie"].
Modele Generatywne (np. GPT):
Jak działa?: Używasz potężnego modelu generatywnego i dajesz mu polecenie (prompt) typu: "Oto zdanie: '[...]' Wygeneruj 5 alternatywnych wersji tego zdania, które mają to samo znaczenie."
To jest najpotężniejsza metoda, ponieważ daje najbardziej płynne, gramatyczne i kreatywne parafrazy, a do tego pozwala na dużą kontrolę nad procesem (jak omawialiśmy wcześniej).
Strategia i Miejsce w Twoim Projekcie
Cel: Twoim głównym celem jest zbalansowanie klas. Zidentyfikuj etykiety, które mają najmniej przykładów (np. poniżej 100). To na nich powinieneś skupić swoje wysiłki augmentacyjne. Nie ma sensu augmentować klasy, która ma już 200 przykładów.
Miejsce w Pipeline: Augmentację stosujesz na swoim "czystym" zbiorze treningowym (3k ręcznych etykiet + ewentualnie najlepsze etykiety ze słabej superwizji). Robisz to przed finalnym treningiem modelu.
Dane -> Słaba Superwizja -> **Augmentacja (na rzadkich klasach)** -> Trenowanie Finalnego Modelu (BERT)
Wybór Metody:
Dobry Punkt Startowy: Back Translation. Jest stosunkowo łatwe do zaimplementowania na dużą skalę (przy użyciu gotowych API lub modeli) i daje znacznie lepsze wyniki niż EDA.
Najlepsza Jakość (Rekomendowane): Modele Generatywne (GPT). Ponieważ masz 200 podobnych etykiet, kluczowe jest zachowanie niuansów znaczeniowych. Tylko modele generatywne z odpowiednim promptem dają Ci taką kontrolę. Koszt API może być czynnikiem, ale jakość wygenerowanych danych jest tego warta.
Kontrola Jakości: To jest niezwykle ważne. Każda metoda augmentacji może wprowadzić szum lub całkowicie zmienić znaczenie zdania, co mogłoby unieważnić oryginalną etykietę.
Filtrowanie Semantyczne: Po wygenerowaniu nowego zdania, użyj modelu do osadzania (np. Sentence-BERT), aby obliczyć podobieństwo semantyczne między zdaniem oryginalnym a zaaugmentowanym. Odrzuć te pary, których podobieństwo jest poniżej pewnego progu (np. 0.8).
Przegląd Ręczny: Przynajmniej na początku, przejrzyj losową próbkę wygenerowanych zdań, aby upewnić się, że Twoja metoda działa poprawnie.
Prawidłowo zastosowana augmentacja danych może być jednym z najbardziej efektywnych kosztowo sposobów na znaczące podniesienie wydajności Twojego modelu, zwłaszcza w kontekście tak dużego imbalansu klas, z jakim się mierzysz.
