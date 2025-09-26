# =================================================================================
# SCRIPT: Few-Shot Learning with SetFit to Reliably Label Data and Correct Relevance
# =================================================================================

# Step 0: Install necessary packages.
# This ensures all required libraries are available.
# !pip install -q pandas==2.2.2 torch==2.3.1 sentence-transformers==3.0.1 setfit==2.0.1 scikit-learn==1.5.0 datasets==2.19.2

import pandas as pd
import numpy as np
import torch
import sys
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

# Import SetFit components, which are essential for this approach
from setfit import SetFitModel, SetFitTrainer

print("--- Starting: Few-Shot Learning to Label Unlabeled Data and Correct Relevance ---")

# --- CONFIGURATION (ACTION REQUIRED) ---
# 1. Set the path to your data file.
FILE_PATH = "your_dataset.csv"  # <--- CHANGE THIS TO YOUR FILE PATH

# 2. Confirm your column names.
TEXT_COLUMN = "TEXT"
LABEL_COLUMN = "DEMAND_ID"
RELEVANT_COLUMN = "RELEVANT"

# 3. Set a confidence threshold. THIS IS THE MOST IMPORTANT SETTING.
# Start with a lower value. If you get too many incorrect labels, increase it.
# If you get too few labels, decrease it. A good starting point is 0.6.
CONFIDENCE_THRESHOLD = 0.60

# 4. Set the name for the output file.
OUTPUT_FILE_PATH = "your_dataset_FINAL_LABELED.csv"
# --- END CONFIGURATION ---


# Step 1: Load the dataset
try:
    df = pd.read_csv(FILE_PATH)
    print(f"\n✅ Successfully loaded dataset with {len(df)} rows from '{FILE_PATH}'.")
    # Ensure the relevance column is an integer (0 or 1)
    df[RELEVANT_COLUMN] = df[RELEVANT_COLUMN].astype(int)
except FileNotFoundError:
    print(f"❌ ERROR: The file '{FILE_PATH}' was not found. Please check the path and try again.")
    sys.exit()

# --- VERIFICATION (BEFORE) ---
print("\n--- DATA STATE (BEFORE SCRIPT) ---")
print("Value counts for the 'RELEVANT' column:")
print(df[RELEVANT_COLUMN].value_counts())
print(f"\nNumber of rows with an empty '{LABEL_COLUMN}': {df[LABEL_COLUMN].isnull().sum()}")
# ---

# Step 2: Prepare the Labeled Data for Training
print("\n🔄 Preparing your labeled examples for training the model...")
train_df = df.dropna(subset=[LABEL_COLUMN]).copy()

if len(train_df) < 10: # Check if there is enough data to train
    print("❌ ERROR: Not enough labeled data found to train a model. Please check your LABEL_COLUMN.")
    sys.exit()

# SetFit requires integer labels, so we convert DEMAND_ID strings to numbers
label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df[LABEL_COLUMN])
train_dataset = Dataset.from_pandas(train_df)
print(f"✅ Training data prepared using {len(train_df)} examples for {len(label_encoder.classes_)} unique labels.")

# Step 3: Fine-Tune a SetFit Model on Your Data
print("\n🔄 Training a specialized model on your data. This is the key step...")
model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    column_mapping={TEXT_COLUMN: "text", "label_encoded": "label"}
)
trainer.train()
print("✅ Model training complete. It is now specialized for your data.")

# Step 4: Identify All Unlabeled Rows to Process
# This correctly selects the 97k rows that have no DEMAND_ID.
rows_to_process_df = df[df[LABEL_COLUMN].isnull()].copy()
texts_to_process = rows_to_process_df[TEXT_COLUMN].astype(str).tolist()

if not texts_to_process:
    print("\n✅ Process finished: No unlabeled sentences were found.")
    sys.exit()
else:
    print(f"\n🔍 Found {len(texts_to_process)} unlabeled sentences to predict and potentially correct.")

# Step 5: Predict Labels and Confidence Scores
print("🔄 Predicting labels for all unlabeled data...")
predicted_probs = trainer.model.predict_proba(texts_to_process)
confidence_scores = np.max(predicted_probs, axis=1)
predicted_indices = np.argmax(predicted_probs, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_indices)

# Add results to our temporary dataframe
rows_to_process_df['confidence_score'] = confidence_scores
rows_to_process_df['predicted_label'] = predicted_labels

# Step 6: Filter for High-Confidence Predictions and Update Main DataFrame
# This is the crucial logic that makes the changes.
successful_assignments = rows_to_process_df[rows_to_process_df['confidence_score'] >= CONFIDENCE_THRESHOLD]

print(f"\n📊 Found {len(successful_assignments)} predictions that meet the {CONFIDENCE_THRESHOLD} confidence threshold.")

if not successful_assignments.empty:
    # Identify how many rows will be flipped from RELEVANT=0 to RELEVANT=1
    rows_to_flip_relevance = df.loc[successful_assignments.index]
    flipped_count = (rows_to_flip_relevance[RELEVANT_COLUMN] == 0).sum()

    print(f"🔄 Applying changes: Assigning {len(successful_assignments)} new labels and correcting relevance...")

    # Use the index of successful assignments to update the main DataFrame
    # 1. Assign the new DEMAND_ID
    df.loc[successful_assignments.index, LABEL_COLUMN] = successful_assignments['predicted_label']
    # 2. Correct the RELEVANT flag to 1
    df.loc[successful_assignments.index, RELEVANT_COLUMN] = 1

    print(f"✅ Corrected the '{RELEVANT_COLUMN}' flag from 0 to 1 for {flipped_count} rows.")
else:
    print("\n⚠️ No predictions met the confidence threshold. No changes were made.")
    print("💡 TIP: Try lowering the CONFIDENCE_THRESHOLD at the top of the script if you expect more matches.")

# --- VERIFICATION (AFTER) ---
print("\n--- DATA STATE (AFTER SCRIPT) ---")
print("Value counts for the 'RELEVANT' column (should now be updated):")
print(df[RELEVANT_COLUMN].value_counts())
print(f"\nNumber of rows with an empty '{LABEL_COLUMN}': {df[LABEL_COLUMN].isnull().sum()}")
# ---

# Step 7: Save the final, corrected result
df.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"\n✅ All done! The updated dataset has been saved to: '{OUTPUT_FILE_PATH}'")
