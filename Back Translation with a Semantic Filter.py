# Oczywiście. Przechodzimy do bardzo praktycznej i potężnej techniki augmentacji: Tłumaczenia w Obie Strony z Filtrem Semantycznym (Back Translation with a Semantic Filter).
# Jest to idealne połączenie kreatywności i kontroli. Back Translation generuje zróżnicowane parafrazy, a Filtr Semantyczny działa jak strażnik jakości, zapewniając, że wygenerowane zdania nie "odpłynęły" zbyt daleko od oryginalnego znaczenia.
# Logika Procesu Krok po Kroku
# Proces składa się z trzech głównych etapów:
# Etap 1: Tłumaczenie "w Przód" (Forward Translation)
# Bierzesz oryginalne zdanie w języku źródłowym (np. polskim).
# Tłumaczysz je na jeden lub więcej języków pośrednich (pivot languages), np. angielski (EN), niemiecki (DE), hiszpański (ES).
# Wskazówka: Używanie języków z różnych rodzin językowych (np. angielski - germańska, hiszpański - romańska, japoński - japońska) często daje bardziej zróżnicowane wyniki.
# Etap 2: Tłumaczenie "w Tył" (Backward Translation)
# Bierzesz przetłumaczone zdania z języków pośrednich.
# Tłumaczysz je z powrotem na język oryginalny (polski).
# Wynik: Otrzymujesz listę zdań-kandydatów, które są parafrazami oryginału. Niektóre będą świetne, inne mogą być dziwne lub całkowicie błędne.
# Etap 3: Filtrowanie Semantyczne (Semantic Filtering)
# To jest kluczowy krok kontroli jakości.
# Dla każdej wygenerowanej parafrazy:
# a. Przekształć oryginalne zdanie w wektor liczbowy (osadzenie) za pomocą modelu językowego (np. Sentence-BERT).
# b. Przekształć wygenerowaną parafrazę w wektor za pomocą tego samego modelu.
# c. Oblicz podobieństwo kosinusowe (cosine similarity) między tymi dwoma wektorami. Wynikiem jest liczba od -1 do 1 (zazwyczaj od 0 do 1), która mówi, jak bardzo podobne są znaczenia tych dwóch zdań.
# d. Porównaj wynik z ustalonym progiem podobieństwa (similarity threshold), np. 0.85.
# e. Jeśli podobieństwo jest powyżej progu, akceptujesz parafrazę i dodajesz ją do swojego zbioru danych. Jeśli jest poniżej, odrzucasz ją jako zbyt zniekształconą.
# Dlaczego to Działa tak Dobrze?
# Różnorodność: Każdy model tłumaczeniowy ma swoje własne "słownictwo" i "rozumienie" składni. Proces tłumaczenia w dwie strony "przepuszcza" oryginalne zdanie przez tę inną "perspektywę", co naturalnie generuje synonimy i zmienia strukturę zdania.
# Bezpieczeństwo: Filtr semantyczny jest Twoją siatką bezpieczeństwa. Chroni Cię przed największym ryzykiem Back Translation – dryfem semantycznym (semantic drift), czyli sytuacją, w której znaczenie zostaje "zgubione w tłumaczeniu".
# Kod w Pythonie
# Do implementacji potrzebne będą biblioteki do tłumaczenia i do osadzania zdań.
# Tłumaczenie: Możemy użyć biblioteki transformers, która daje dostęp do darmowych, wysokiej jakości modeli tłumaczeniowych z Hugging Face (np. od grupy Helsinki-NLP). To lepsze niż poleganie na zewnętrznym API.
# Osadzanie: Użyjemy sentence-transformers do filtrowania.
# Zainstaluj potrzebne biblioteki:
# pip install transformers sentence-transformers torch accelerate sentencepiece
# (accelerate jest zalecany, sentencepiece jest potrzebny dla wielu modeli tłumaczeniowych)

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# --- Konfiguracja Modeli (załadować raz, używać wielokrotnie) ---

# Sprawdź, czy dostępne jest GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używane urządzenie: {device}")

# 1. Modele do Tłumaczenia (Helsinki-NLP)
print("Ładowanie modeli tłumaczeniowych...")
# Model z polskiego na angielski
tokenizer_pl_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-pl-en")
model_pl_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-pl-en").to(device)
# Model z angielskiego na polski
tokenizer_en_pl = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-pl")
model_en_pl = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-pl").to(device)
print("Modele tłumaczeniowe załadowane.")

# 2. Model do Osadzania (Sentence-BERT)
print("Ładowanie modelu do osadzania...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
print("Model do osadzania załadowany.")


# --- Funkcja do Augmentacji ---

def augment_with_back_translation(original_text: str, similarity_threshold: float = 0.85) -> str | None:
    """
    Wykonuje augmentację tekstu za pomocą Back Translation i zwraca wynik,
    jeśli przejdzie on przez filtr semantyczny. W przeciwnym razie zwraca None.
    """
    try:
        # --- Etap 1: Tłumaczenie "w Przód" (PL -> EN) ---
        tokenized_text = tokenizer_pl_en.prepare_seq2seq_batch([original_text], return_tensors='pt').to(device)
        translation_ids = model_pl_en.generate(**tokenized_text)
        translated_text_en = tokenizer_pl_en.batch_decode(translation_ids, skip_special_tokens=True)[0]

        # --- Etap 2: Tłumaczenie "w Tył" (EN -> PL) ---
        tokenized_text_en = tokenizer_en_pl.prepare_seq2seq_batch([translated_text_en], return_tensors='pt').to(device)
        back_translation_ids = model_en_pl.generate(**tokenized_text_en)
        paraphrase = tokenizer_en_pl.batch_decode(back_translation_ids, skip_special_tokens=True)[0]
        
        # --- Etap 3: Filtrowanie Semantyczne ---
        # Unikajmy porównywania identycznych zdań
        if paraphrase.lower() == original_text.lower():
            return None

        # Obliczanie osadzeń
        original_embedding = embedding_model.encode(original_text, convert_to_tensor=True)
        paraphrase_embedding = embedding_model.encode(paraphrase, convert_to_tensor=True)
        
        # Obliczanie podobieństwa kosinusowego
        cosine_score = util.pytorch_cos_sim(original_embedding, paraphrase_embedding).item()
        
        # Sprawdzenie progu
        if cosine_score >= similarity_threshold:
            print(f"  -> AKCEPTACJA (Podobieństwo: {cosine_score:.4f}): '{paraphrase}'")
            return paraphrase
        else:
            print(f"  -> ODRZUCENIE (Podobieństwo: {cosine_score:.4f}): '{paraphrase}'")
            return None
            
    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania: {e}")
        return None

# --- Użycie ---
original_sentence = "Ten tusz do rzęs jest wodoodporny i nie osypuje się."
print(f"Oryginalne zdanie: '{original_sentence}'")
print("Generowanie parafrazy...")

augmented_sentence = augment_with_back_translation(original_sentence, similarity_threshold=0.8)

if augmented_sentence:
    print(f"\nZaakceptowana augmentacja: '{augmented_sentence}'")
else:
    print("\nNie udało się wygenerować zaakceptowanej augmentacji.")

print("-" * 20)

# Przykład, który może zostać odrzucony
original_sentence_2 = "To jest absolutny hit sprzedaży w naszej drogerii."
print(f"Oryginalne zdanie: '{original_sentence_2}'")
print("Generowanie parafrazy...")
augmented_sentence_2 = augment_with_back_translation(original_sentence_2, similarity_threshold=0.9)

# Strategia i Najlepsze Praktyki
# Dobór Progu (similarity_threshold): To najważniejszy parametr do strojenia.
# Wysoki próg (0.9 - 0.95): Otrzymasz bardzo bezpieczne parafrazy, które są bliskie oryginałowi. Mniejsza różnorodność, ale mniejsze ryzyko wprowadzenia szumu.
# Średni próg (0.8 - 0.9): Dobry kompromis między bezpieczeństwem a różnorodnością. Zazwyczaj najlepszy punkt startowy.
# Niski próg (0.7 - 0.8): Większa różnorodność, ale też znacznie większe ryzyko, że zaakceptujesz zdanie, które zmieniło znaczenie.
# Wiele Języków Pośrednich: Aby wygenerować więcej niż jedną parafrazę dla jednego zdania, stwórz listę par modeli (np. pl-en/en-pl, pl-de/de-pl) i wykonaj proces dla każdej pary. To da Ci bogatszy zbiór kandydatów.
# Wydajność: Przetwarzanie 90k zdań w ten sposób może być wolne. Przetwarzaj dane w partiach (batch processing), aby lepiej wykorzystać GPU. Zarówno modele tłumaczeniowe, jak i sentence-transformers są znacznie szybsze, gdy działają na wielu zdaniach naraz.
# Zastosowanie: Ta technika jest idealna dla Twojej strategii Ukierunkowanej Augmentacji Danych (Focused Data Augmentation). Użyj jej, aby "rozmnożyć" przykłady dla Twoich rzadkich klas, zapewniając jednocześnie, że wygenerowane dane są wysokiej jakości i trzymają się oryginalnego tematu.
