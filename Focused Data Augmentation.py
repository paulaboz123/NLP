# Oczywiście. Przechodzimy teraz do Ukierunkowanej Augmentacji Danych (Focused Data Augmentation). Jest to bardziej strategia niż konkretna technika. Zamiast augmentować dane "na ślepo", robimy to w sposób przemyślany i celowy, aby rozwiązać konkretne problemy, które zdiagnozowaliśmy w naszym modelu.
# W Twoim przypadku, dwa główne problemy, które można zaatakować za pomocą tej strategii, to:
# Ekstremalnie niski recall (10%) dla klasy relevant.
# Ogromny imbalans klas wśród 200 etykiet (niektóre mają 20 przykładów, inne 200).
# Główna Idea: Wzmocnij To, Czego Model Nie Rozumie
# Standardowa augmentacja mówi: "Weźmy nasze dane i stwórzmy ich więcej". Ukierunkowana augmentacja mówi: "Sprawdźmy, gdzie nasz model popełnia najwięcej błędów i stwórzmy więcej danych, które są podobne do tych błędów, aby zmusić model do nauki".
# To jak z korepetycjami. Zamiast przerabiać cały podręcznik od nowa, dobry korepetytor skupia się na tych typach zadań, z którymi uczeń radzi sobie najsłabiej.
# Strategia 1: Ukierunkowanie na Zwiększenie Recallu Klasy relevant
# Twój model przepuszcza 90% relewantnych zdań. Oznacza to, że jest "zbyt konserwatywny" i nauczył się, że większość danych jest nierelewantna, więc bezpieczniej jest zgadywać not_relevant. Musimy go "ośmielić", pokazując mu ogromną różnorodność tego, co może być relewantne.
# Jak to zrobić?
# Zidentyfikuj Zbiór Relewantny: Weź wszystkie swoje 3000 oznaczonych zdań. To jest Twoja "złota" pula danych relewantnych.
# Zastosuj Agresywną Augmentację: Dla każdego z tych 3000 zdań, wygeneruj nie jedną, ale np. 5-10 zróżnicowanych wariacji. Użyj do tego najlepszych dostępnych technik:
# Back Translation: Użyj wielu języków pośrednich (np. PL -> EN -> PL, PL -> DE -> PL, PL -> ES -> PL), aby uzyskać dużą różnorodność składniową.
# Augmentacja z GPT: Użyj promptów, które zachęcają do kreatywności, np.:
# "Przepisz to zdanie w bardziej potocznym stylu, zachowując jego znaczenie."
# "Napisz to zdanie tak, jakby było pytaniem na forum internetowym."
# "Stwórz 5 parafraz tego zdania, używając różnych synonimów dla kluczowych produktów."
# Stwórz Zbalansowany Zbiór dla "Bramkarza": Jeśli używasz architektury z osobnym klasyfikatorem binarnym, ta technika jest idealna. Twój oryginalny zbiór to 3k (relevant) vs 87k (not_relevant). Po augmentacji możesz stworzyć nowy zbiór treningowy dla "Bramkarza", który będzie znacznie bardziej zbalansowany, np. 30 000 (3k * 10 augmentacji) przykładów relevant vs losowa próbka 30 000 przykładów not_relevant.
# Efekt: Trenując na tak zbalansowanym zbiorze, model przestanie być uprzedzony do klasy not_relevant i znacznie chętniej będzie przewidywał relevant, co bezpośrednio podniesie jego recall.
# Strategia 2: Ukierunkowanie na Wyrównanie Imbalansu Między Etykietami
# Twój model jest świetny w rozpoznawaniu powder (200 przykładów), ale słaby w liquid powder (20 przykładów). Dzieje się tak, ponieważ widział 10x więcej przykładów tej pierwszej. Musimy sztucznie wyrównać te proporcje.
# Jak to zrobić?
# Zdefiniuj Cel: Ustal docelową liczbę przykładów dla każdej klasy, np. "chcę, aby każda klasa miała co najmniej 150 przykładów".
# Oblicz Deficyt: Dla każdej klasy oblicz, ile przykładów jej brakuje do osiągnięcia celu.
# powder: 150 (cel) - 200 (istniejące) = -50. Nie robisz nic.
# mascara: 150 (cel) - 100 (istniejące) = 50. Musisz wygenerować 50 nowych przykładów.
# liquid powder: 150 (cel) - 20 (istniejące) = 130. Musisz wygenerować 130 nowych przykładów.
# Zastosuj Proporcjonalną Augmentację: Dla każdej klasy, która ma deficyt, zastosuj augmentację, aby wygenerować brakującą liczbę przykładów.
# Dla mascara (100 przykładów, potrzeba 50), wystarczy, że dla co drugiego zdania wygenerujesz jedną nową wariację (100 * 0.5 = 50).
# Dla liquid powder (20 przykładów, potrzeba 130), musisz dla każdego zdania wygenerować średnio 6-7 nowych wariacji (20 * 6.5 = 130).
# Użyj Najlepszych Technik: Ponieważ rzadkie klasy są często bardziej specyficzne i trudniejsze, użyj najlepszych dostępnych metod (GPT z kontrolą jakości), aby zapewnić, że wygenerowane dane nie zmienią oryginalnego, subtelnego znaczenia.
# Kod Koncepcyjny: Logika Ukierunkowanej Augmentac

import pandas as pd

# Załóżmy, że mamy funkcję `augment_text(text, n_variations)`
# która zwraca listę n wariacji danego tekstu.
# W praktyce byłaby to funkcja wołająca API GPT lub Back Translation.

# --- Przykład 2: Wyrównywanie Imbalansu Etykiet ---

# Dane wejściowe
data = {
    'text': ['puder 1', 'puder 2', 'puder 3', 'mascara 1', 'liquid_powder 1'],
    'label': ['powder', 'powder', 'powder', 'mascara', 'liquid_powder']
}
df = pd.DataFrame(data)

TARGET_COUNT = 3
augmented_data = []

# Oblicz liczebność każdej klasy
class_counts = df['label'].value_counts()

for label, count in class_counts.items():
    if count < TARGET_COUNT:
        # Oblicz, ile przykładów trzeba dogenerować
        n_to_generate = TARGET_COUNT - count
        
        # Pobierz wszystkie istniejące zdania dla tej rzadkiej klasy
        sentences_to_augment = df[df['label'] == label]['text'].tolist()
        
        # Oblicz, ile wariacji na zdanie trzeba stworzyć
        # np. ceil(10 do wygenerowania / 3 istniejące) -> 4 wariacje na zdanie
        n_variations_per_sentence = int(np.ceil(n_to_generate / count))
        
        print(f"Klasa '{label}': Brakuje {n_to_generate} przykładów. Generowanie {n_variations_per_sentence} wariacji na zdanie.")
        
        generated_count = 0
        for text in sentences_to_augment:
            if generated_count >= n_to_generate:
                break
            
            # Załóżmy, że augment_text zwraca listę nowych zdań
            new_sentences = augment_text(text, n_variations=n_variations_per_sentence)
            
            for new_sentence in new_sentences:
                if generated_count >= n_to_generate:
                    break
                augmented_data.append({'text': new_sentence, 'label': label})
                generated_count += 1

# Połącz oryginalne dane z nowo wygenerowanymi
if augmented_data:
    df_augmented = pd.DataFrame(augmented_data)
    df_final = pd.concat([df, df_augmented], ignore_index=True)
else:
    df_final = df

print("\nFinalna liczebność klas:")
print(df_final['label'].value_counts())

# Podsumowanie i Miejsce w Twoim Projekcie
# Ukierunkowana augmentacja to inteligentna warstwa strategii nałożona na techniki augmentacji. Zamiast pytać "Jak mogę stworzyć więcej danych?", pytasz "Jakie dane muszę stworzyć, aby rozwiązać konkretny problem mojego modelu?".
# Miejsce w Pipeline: Jest to krok, który wykonujesz na swoim zbiorze treningowym przed treningiem lub retrenowaniem modelu. Często jest to proces iteracyjny: trenujesz model, analizujesz jego błędy (np. macierz konfuzji, niski recall dla pewnych klas), a następnie stosujesz ukierunkowaną augmentację, aby "naprawić" te słabości i trenujesz ponownie.
# W połączeniu z Innymi Technikami: Działa doskonale w synergii. Na przykład, po użyciu Osobnego Klasyfikatora Binarnego, możesz zastosować ukierunkowaną augmentację na klasie relevant, aby jeszcze bardziej poprawić jego recall.
# To podejście pozwala na znacznie bardziej efektywne wykorzystanie zasobów i prowadzi do budowy modeli, które są nie tylko ogólnie dokładne, ale także silne w obszarach, które są dla Ciebie najważniejsze.
