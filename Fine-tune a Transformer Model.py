# Oczywiście. Przechodzimy do kroku, który łączy Twoje starannie przygotowane dane z surową mocą obliczeniową nowoczesnego NLP: Dostrajanie (Fine-tuning) Modelu Transformerowego.
# To jest obecnie złoty standard dla prawie każdego zadania związanego z rozumieniem tekstu. Jeśli poprzednie kroki (słaba superwizja, augmentacja) były o przygotowaniu najlepszych składników, ten krok jest o użyciu najnowocześniejszego piekarnika, aby uzyskać idealny wypiek.
# Intuicja: Zatrudnienie Genialnego, Przesadnie Wykwalifikowanego Stażysty
# Wyobraź sobie, że zatrudniasz stażystę. Ale ten stażysta (model transformerowy, np. BERT) nie jest zwykły. Zanim przyszedł do Twojej firmy, przeczytał i zrozumiał całą Wikipedię, tysiące książek i ogromną część internetu (to jest etap pre-treningu).
# Co on już umie?: Rozumie gramatykę, składnię, niuanse, sarkazm, a co najważniejsze – kontekst. Wie, że "zamek" w zdaniu o drzwiach to coś innego niż "zamek" w zdaniu o królu.
# Czego nie umie?: Nie zna specyfiki Twojej firmy (Twojej domeny o kosmetykach). Nie wie, co to jest liquid powder i czym różni się od setting powder.
# Twoje zadanie (Fine-tuning): Dajesz mu do przeczytania wszystkie swoje wewnętrzne dokumenty (Twój zbiór treningowy z 3k+ przykładami) i mówisz: "Teraz zastosuj całą swoją ogromną wiedzę o języku, aby nauczyć się klasyfikować tylko tego typu zdania".
# Dostrajanie to proces adaptacji tej gigantycznej, ogólnej wiedzy do Twojego bardzo specyficznego, wąskiego zadania.
# Dlaczego to jest absolutnie kluczowe dla Twojego projektu?
# Zrozumienie Kontekstu: Tylko model transformerowy jest w stanie skutecznie odróżnić Twoje 200 podobnych etykiet. Zrozumie, że w zdaniu "ten puder w płynie matuje cerę" słowa "w płynie" całkowicie zmieniają znaczenie "pudru". Prostsze modele (jak TF-IDF) tego nie potrafią.
# Uczenie Transferowe (Transfer Learning): To jest serce tej metody. Model "przenosi" wiedzę z pre-treningu na Twoje zadanie. Dzięki temu nawet dla Twojej etykiety z 20 przykładami (liquid powder) model nie uczy się od zera. On już wie, co znaczą słowa "liquid" i "powder" – musi się tylko nauczyć ich specyficznej kombinacji w Twoim kontekście.
# Wydajność State-of-the-Art: Ta metoda konsekwentnie osiąga najwyższe wyniki w benchmarkach NLP.
# Proces Dostrajania Krok po Kroku
# Wybór Modelu Bazowego: Wybierasz wstępnie wytrenowany model z "hubu" (np. Hugging Face Hub). Popularne wybory to BERT, RoBERTa, DistilBERT (szybszy), lub modele wielojęzyczne, jeśli Twoje dane nie są po angielsku.
# Dodanie "Głowicy" Klasyfikacyjnej: Do "ciała" transformera (które rozumie tekst) dodajesz małą, specyficzną dla zadania warstwę na końcu. Dla Twojego problemu wieloetykietowego, będzie to warstwa z 200 neuronami i aktywacją sigmoid.
# Tokenizacja: Dzielisz swoje teksty na tokeny (słowa/części słów), które model rozumie, i konwertujesz je na liczby.
# Trening: "Pokazujesz" modelowi swoje dane partiami (batches). Model dokonuje predykcji, porównuje je z prawdziwymi etykietami, oblicza błąd (stratę) i lekko modyfikuje swoje wagi, aby następnym razem popełnić mniejszy błąd. To właśnie jest "dostrajanie".
# Ewaluacja i Predykcja: Po zakończeniu treningu używasz dostrojonego modelu do klasyfikacji nowych, niewidzianych wcześniej zdań.
# Kod: Dostrajanie Modelu do Klasyfikacji Wieloetykietowej (z Hugging Face)
# Do tego zadania biblioteka transformers od Hugging Face jest absolutnym standardem. Upraszcza ona cały proces. Zainstaluj potrzebne biblioteki: pip install transformers datasets scikit-learn torch.

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# --- Krok 1: Przygotowanie Danych ---
# Przykładowy DataFrame naśladujący Twoje dane
data = {
    'text': [
        'Ta nowa pomadka ma piękny, matowy kolor.',
        'Mój ulubiony tusz do rzęs świetnie wydłuża.',
        'Ten podkład i korektor idealnie kryją niedoskonałości.',
        'Nowa szminka z drobinkami brokatu.',
        'Wodoodporna maskara to must-have na lato.',
        'Fluid i puder w jednym produkcie.'
    ],
    # Etykiety jako listy stringów
    'labels': [
        ['lipstick'], ['mascara'], ['foundation', 'concealer'],
        ['lipstick'], ['mascara'], ['foundation', 'powder']
    ]
}
df = pd.DataFrame(data)

# Pobierz wszystkie unikalne etykiety ze zbioru danych
all_labels = sorted(list(set(label for sublist in df['labels'] for label in sublist)))
label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for label, i in label2id.items()}

# Konwersja etykiet tekstowych na format multi-hot vector (np. [0, 1, 1, 0, ...])
# To jest kluczowy krok dla klasyfikacji wieloetykietowej
mlb = MultiLabelBinarizer(classes=all_labels)
y = mlb.fit_transform(df['labels'])
# Dodajemy sformatowane etykiety do DataFrame jako listę floatów
df['labels_vector'] = [np.array(row, dtype=np.float32) for row in y]

# Konwersja pandas DataFrame na Dataset Hugging Face
dataset = Dataset.from_pandas(df)


# --- Krok 2: Tokenizacja ---
model_name = "bert-base-multilingual-cased" # Dobry model, jeśli dane są po polsku
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_data(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

dataset = dataset.map(tokenize_data, batched=True)
# Zmieniamy nazwę kolumny, aby pasowała do oczekiwań modelu
dataset = dataset.rename_column("labels_vector", "labels")


# --- Krok 3: Załadowanie Modelu ---
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(all_labels),
    problem_type="multi_label_classification", # KLUCZOWY PARAMETR!
    id2label=id2label,
    label2id=label2id
)
# Parametr 'problem_type' automatycznie konfiguruje model:
# - Używa aktywacji Sigmoid na wyjściu (zamiast Softmax).
# - Używa Binary Cross-Entropy jako funkcji straty.

# --- Krok 4: Konfiguracja i Trening ---
# Zdefiniuj metryki do ewaluacji
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # Zastosuj sigmoid, a następnie próg 0.5, aby uzyskać finalne predykcje binarne
    sigmoid_preds = 1 / (1 + np.exp(-preds))
    binary_preds = (sigmoid_preds > 0.5).astype(int)
    
    f1 = f1_score(y_true=p.label_ids, y_pred=binary_preds, average='samples')
    accuracy = accuracy_score(y_true=p.label_ids, y_pred=binary_preds)
    return {"f1_samples": f1, "accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5, # Mały learning rate jest kluczowy przy fine-tuningu
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset, # W praktyce tu będzie Twój zbiór treningowy
    # eval_dataset=dataset, # Tu będzie zbiór walidacyjny
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Rozpocznij dostrajanie!
# trainer.train() 
# (Powyższa linia jest zakomentowana, bo wymaga środowiska z GPU i pełnego zbioru danych)
print("Konfiguracja do dostrajania modelu Transformer jest gotowa.")
print(f"Model zostanie wytrenowany na {len(all_labels)} etykietach: {all_labels}")


# Najlepsze Praktyki i Co Dalej?
# Zarządzanie Zasobami: Trenowanie transformerów wymaga GPU. Bez niego proces będzie trwał ekstremalnie długo. Możesz użyć usług takich jak Google Colab (oferuje darmowe GPU) lub platform chmurowych.
# Hiperparametry: Kluczowe parametry do strojenia to learning_rate (zazwyczaj bardzo mały, np. 2e-5 do 5e-5), batch_size (tak duży, jak zmieści się w pamięci GPU) i num_train_epochs (zazwyczaj 2-4, więcej może prowadzić do przeuczenia).
# Walka z Imbalansem: Nawet z transformerem, imbalans klas jest problemem. Rozważ użycie Focal Loss (można zaimplementować własną klasę Trainer z niestandardową funkcją straty) lub ważenie klas, aby model zwracał większą uwagę na rzadkie etykiety.
# Dostrajanie modelu transformerowego to najbardziej pracochłonny obliczeniowo, ale i najbardziej obiecujący krok. To właśnie tutaj cała Twoja praca nad przygotowaniem danych przekłada się na wysoką dokładność i zdolność modelu do radzenia sobie ze złożonością Twojego zadania.
