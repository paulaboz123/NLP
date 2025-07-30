import pandas as pd
import numpy as np
import torch
import evaluate
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset

# Ustawienia wyświetlania dla Pandas
pd.set_option('display.max_colwidth', 200)

print("Biblioteki zostały zaimportowane.")
Use code with caution.
Python
Komórka 2: Konfiguracja (EDYTUJ TUTAJ)
Jedyne miejsce, które musisz edytować.
Generated python
# --- EDYTUJ TE WARTOŚCI ---

# Ścieżka do Twojego FINALNEGO ZBIORU TRENINGOWEGO (wynik poprzedniego notatnika)
TRAINING_DATA_PATH = './final_training_data_for_transformer.csv'

# Ścieżka do LOKALNEGO folderu z modelem bazowym (tym samym, co do tworzenia embeddingów)
BASE_MODEL_PATH = '/sciezka/do/twojego/modelu/lokalnego/'

# Folder, w którym będą zapisywane wyniki (modele, checkpointy)
OUTPUT_DIR = './training_output'

# Nazwy kolumn w Twoim pliku CSV
TEXT_INPUT_COLUMN = 'text_input'
LABEL_COLUMN = 'is_match'

# --- PARAMETRY PROCESU (można dostosować) ---

# Procent danych przeznaczony na zbiór walidacyjny
TEST_SIZE = 0.2

# Liczba prób, które wykona algorytm tuningu hiperparametrów.
# 10-20 to dobry start.
NUM_HYPERPARAMETER_TRIALS = 15

# --- Konfiguracja techniczna ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Konfiguracja zakończona.")
print(f"Używane urządzenie: {device.upper()}")
Use code with caution.
Python
Komórka 3: Wczytanie danych i podział na zbiór treningowy/walidacyjny
Generated python
print("--- Etap 1: Wczytywanie i przygotowanie danych ---")

# Wczytanie finalnego zbioru
df = pd.read_csv(TRAINING_DATA_PATH)

# Podział na zbiór treningowy i walidacyjny
# Używamy stratify, aby zachować ten sam rozkład klas w obu zbiorach
train_df, eval_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=df[LABEL_COLUMN]
)

print(f"Rozmiar zbioru treningowego: {len(train_df)}")
print(f"Rozmiar zbioru walidacyjnego: {len(eval_df)}")
print("\nRozkład etykiet w zbiorze treningowym:")
print(train_df[LABEL_COLUMN].value_counts(normalize=True))
Use code with caution.
Python
Komórka 4: Tokenizer i klasa Dataset
Przygotowujemy dane do formatu, który rozumie model Transformer.
Generated python
print("\n--- Etap 2: Przygotowanie Tokenizera i Datasetu ---")

# Wczytanie tokenizera
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# Definicja własnej klasy Dataset
class PairClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Utworzenie obiektów Dataset
train_dataset = PairClassificationDataset(
    texts=train_df[TEXT_INPUT_COLUMN].tolist(),
    labels=train_df[LABEL_COLUMN].tolist(),
    tokenizer=tokenizer
)

eval_dataset = PairClassificationDataset(
    texts=eval_df[TEXT_INPUT_COLUMN].tolist(),
    labels=eval_df[LABEL_COLUMN].tolist(),
    tokenizer=tokenizer
)

print("Tokenizer i obiekty Dataset gotowe.")
Use code with caution.
Python
Komórka 5: Definicja metryk do ewaluacji
Definiujemy funkcję, która będzie obliczać F1-score, precyzję i czułość podczas walidacji.
Generated python
print("\n--- Etap 3: Definicja metryk ---")

# Wczytanie metryk z biblioteki evaluate
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1 = f1_metric.compute(predictions=predictions, references=labels, pos_label=1)["f1"]
    precision = precision_metric.compute(predictions=predictions, references=labels, pos_label=1)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, pos_label=1)["recall"]
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

print("Funkcja do obliczania metryk gotowa.")
Use code with caution.
Python
Komórka 6: Automatyczny Tuning Hiperparametrów
To jest serce optymalizacji. Używamy Trainer w połączeniu z optuna, aby automatycznie przetestować różne kombinacje hiperparametrów i znaleźć najlepszą.
Generated python
print("\n--- Etap 4: Automatyczny Tuning Hiperparametrów ---")

# Funkcja inicjująca model. Jest wymagana przez hyperparameter_search,
# aby każda próba zaczynała z nowym, nieprzetrenowanym modelem.
def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=2,
        return_dict=True
    )

# Podstawowe argumenty treningowe
training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/hyperparam_search",
    disable_tqdm=True, # Wyłączamy paski postępu, aby nie zaśmiecać logów
)

# Inicjalizacja Trainera
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Definicja przestrzeni poszukiwań dla Optuny
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    }

# Uruchomienie poszukiwania
print(f"Uruchamianie poszukiwania najlepszych hiperparametrów ({NUM_HYPERPARAMETER_TRIALS} prób)...")
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=NUM_HYPERPARAMETER_TRIALS
)

print("\n--- Poszukiwanie zakończone! ---")
print(f"Najlepszy wynik (F1-score): {best_run.objective}")
print("Najlepsze hiperparametry:")
print(best_run.hyperparameters)
Use code with caution.
Python
Komórka 7: Trening finalnego modelu z najlepszymi parametrami
Teraz, gdy znamy najlepszą konfigurację, trenujemy model od nowa, używając tych optymalnych ustawień.
Generated python
print("\n--- Etap 5: Trening finalnego modelu z najlepszymi hiperparametrami ---")

# Ustawienie najlepszych parametrów znalezionych w poprzednim kroku
best_params = best_run.hyperparameters

final_training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/final_model",
    # Użycie najlepszych parametrów
    learning_rate=best_params["learning_rate"],
    num_train_epochs=best_params["num_train_epochs"],
    per_device_train_batch_size=best_params["per_device_train_batch_size"],
    weight_decay=best_params["weight_decay"],
    
    # Standardowe argumenty
    evaluation_strategy="epoch", # Ocena po każdej epoce
    save_strategy="epoch",       # Zapisywanie modelu po każdej epoce
    load_best_model_at_end=True, # Na końcu wczytaj najlepszy model
    metric_for_best_model="f1",  # Użyj F1 jako metryki do wyboru najlepszego modelu
    logging_steps=100,
    fp16=torch.cuda.is_available(), # Użyj precyzji mieszanej (szybszy trening na GPU)
)

# Inicjalizacja finalnego Trenera
final_trainer = Trainer(
    model=model_init(None), # Inicjalizujemy nowy model
    args=final_training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Uruchomienie finalnego treningu
print("\nUruchamianie finalnego treningu...")
final_trainer.train()

print("\n--- Trening zakończony! ---")
Use code with caution.
Python
Komórka 8: Ostateczna ewaluacja i zapisanie modelu
Na koniec sprawdzamy, jak nasz najlepszy model radzi sobie na zbiorze walidacyjnym i zapisujemy go do użytku w przyszłości.
Generated python
print("\n--- Etap 6: Finalna ewaluacja i zapisanie modelu ---")

# Ewaluacja najlepszego modelu
eval_results = final_trainer.evaluate()

print("\nWyniki ewaluacji na zbiorze walidacyjnym:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")
    
# Zapisanie finalnego, najlepszego modelu
final_model_path = f"{OUTPUT_DIR}/best_tuned_model"
final_trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"\nNajlepszy model został zapisany w folderze: {final_model_path}")
print("\n--- PROCES TRENINGU I OPTYMALIZACJI ZAKOŃCZONY POMYŚLNIE! ---")




# --- SEKCJA DIAGNOSTYCZNA ---
print("\n--- ROZPOCZYNANIE DIAGNOSTYKI ZMIENNEJ 'negative_embeddings_cleaned' ---")
try:
    # 1. Sprawdzenie typu i kształtu
    print(f"Typ obiektu: {type(negative_embeddings_cleaned)}")
    # Sprawdzamy czy to jest numpy array - jeśli tak, ma atrybut .shape
    if isinstance(negative_embeddings_cleaned, np.ndarray):
        print(f"Kształt macierzy (shape): {negative_embeddings_cleaned.shape}")
        print(f"Typ danych w macierzy (dtype): {negative_embeddings_cleaned.dtype}")
        
        # 2. GŁĘBOKA DIAGNOSTYKA: Sprawdzenie, czy w środku nie ma "śmieci"
        # Jeśli dtype to 'object', to prawie na pewno mamy problem
        if negative_embeddings_cleaned.dtype == 'object':
            print("\nOSTRZEŻENIE: dtype to 'object'! To sugeruje mieszane typy danych w macierzy.")
            non_numeric_found = False
            for i, emb in enumerate(negative_embeddings_cleaned):
                # Sprawdzamy, czy element jest listą lub numpy array, a nie np. stringiem
                if not isinstance(emb, (np.ndarray, list)):
                    print(f"!!! ZNALEZIONO PROBLEM !!! Wiersz {i} nie jest wektorem. Jego typ to: {type(emb)}")
                    print(f"Treść problematycznego wiersza: {emb}")
                    print(f"Oryginalny tekst, który go wygenerował: {texts_to_encode[i]}")
                    non_numeric_found = True
                    break # Przerywamy po znalezieniu pierwszego problemu
            if not non_numeric_found:
                print("Nie znaleziono bezpośrednich 'śmieci', ale dtype 'object' jest niepokojący.")
                
    else:
        print("OSTRZEŻENIE: Wynik z model.encode() nie jest standardową macierzą numpy!")

except Exception as e:
    print(f"Błąd podczas diagnostyki: {e}")

print("--- ZAKOŃCZONO DIAGNOSTYKĘ ---")
# --- KONIEC SEKCJI DIAGNOSTYCZNEJ ---
