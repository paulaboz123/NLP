# =================================================================================
# SCRIPT: Few-Shot Learning with SetFit to Label Data and Correct Relevance
# =================================================================================

# Step 0: Install necessary packages.
# SetFit is now included as it is the core of this new approach.
# !pip install -q pandas==2.2.2 torch==2.3.1 sentence-transformers==3.0.1 setfit==2.0.1 scikit-learn==1.5.0 datasets==2.19.2

import pandas as pd
import numpy as np
import torch
import sys
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

# Import SetFit components
from setfit import SetFitModel, SetFitTrainer

print("--- Few-Shot Learning Labeling and Correction Process ---")
print("This script will train a model on your 3k labeled examples to label the rest.")

# --- CONFIGURATION (ACTION REQUIRED) ---
# 1. Set the path to your data file.
FILE_PATH = "your_dataset.csv"  # <--- CHANGE THIS TO YOUR FILE PATH

# 2. Confirm your column names.
TEXT_COLUMN = "TEXT"
LABEL_COLUMN = "DEMAND_ID"
RELEVANT_COLUMN = "RELEVANT"

# 3. Set a confidence threshold for assigning a label and correcting relevance.
# A higher value (e.g., 0.75 or higher) is recommended for this approach,
# as the fine-tuned model's probabilities are generally more reliable.
CONFIDENCE_THRESHOLD = 0.75

# 4. Set the name for the output file.
OUTPUT_FILE_PATH = "your_dataset_finetuned_labeled.csv"
# --- END CONFIGURATION ---


# Step 1: Load the dataset
try:
    df = pd.read_csv(FILE_PATH)
    print(f"\n✅ Successfully loaded dataset with {len(df)} rows from '{FILE_PATH}'.")
    df[RELEVANT_COLUMN] = df[RELEVANT_COLUMN].astype(int)
except FileNotFoundError:
    print(f"❌ ERROR: The file '{FILE_PATH}' was not found. Please check the path.")
    sys.exit()

# --- VERIFICATION (BEFORE) ---
print("\n--- DATA STATE (BEFORE SCRIPT) ---")
print("Value counts for 'RELEVANT' column (this will be corrected):")
print(df[RELEVANT_COLUMN].value_counts())
print(f"\nNumber of rows with an empty '{LABEL_COLUMN}': {df[LABEL_COLUMN].isnull().sum()}")
# ---

# Step 2: Prepare the Labeled Data for Training
print("\n🔄 Preparing your 3,000 labeled examples for training...")
train_df = df.dropna(subset=[LABEL_COLUMN]).copy()

if len(train_df) == 0:
    print("❌ ERROR: No labeled data found to train the model. Stopping script.")
    sys.exit()

# SetFit requires integer labels, so we encode DEMAND_ID strings to numbers
label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df[LABEL_COLUMN])

# Convert the pandas DataFrame to a Hugging Face Dataset object
train_dataset = Dataset.from_pandas(train_df)
print(f"✅ Training data prepared with {len(train_df)} examples and {len(label_encoder.classes_)} unique labels.")

# Step 3: Fine-Tune the SetFit Model on Your Labeled Data
print("\n🔄 Fine-tuning a new model on your data. This may take a few minutes...")
# Load a pre-trained Sentence Transformer model to be fine-tuned
model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Create a trainer object
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    column_mapping={TEXT_COLUMN: "text", "label_encoded": "label"} # Map your columns to what SetFit expects
)

# Execute the training
trainer.train()
print("✅ Model successfully fine-tuned and specialized for your DEMAND_IDs.")

# Step 4: Prepare the Unlabeled Data for Prediction
# We select ALL unlabeled rows, regardless of their current 'RELEVANT' status
rows_to_process_df = df[df[LABEL_COLUMN].isnull()].copy()
texts_to_process = rows_to_process_df[TEXT_COLUMN].astype(str).tolist()

if not texts_to_process:
    print("\n✅ Process finished: No unlabeled sentences were found to process.")
    sys.exit()
else:
    print(f"\n🔍 Found {len(texts_to_process)} unlabeled sentences to predict.")

# Step 5: Predict Labels and Confidence Scores for Unlabeled Data
print("🔄 Predicting labels for unlabeled data using your new fine-tuned model...")
# Get the predicted label probabilities (confidence scores)
predicted_probs = trainer.model.predict_proba(texts_to_process)

# Find the highest probability and the index of the corresponding label
confidence_scores = np.max(predicted_probs, axis=1)
predicted_indices = np.argmax(predicted_probs, axis=1)

# Convert the predicted integer indices back to your original DEMAND_ID strings
predicted_labels = label_encoder.inverse_transform(predicted_indices)

# Add results to our temporary dataframe
rows_to_process_df['confidence_score'] = confidence_scores
rows_to_process_df['predicted_label'] = predicted_labels

# Filter for assignments that meet our high-confidence threshold
successful_assignments = rows_to_process_df[rows_to_process_df['confidence_score'] >= CONFIDENCE_THRESHOLD].copy()

print(f"✅ Found {len(successful_assignments)} high-confidence predictions to apply.")

# Step 6: Update Main DataFrame with New Labels and Corrected Relevance
if not successful_assignments.empty:
    print("\n🔄 Updating main DataFrame with new labels and corrected 'RELEVANT' flags...")

    # Identify which rows had their 'RELEVANT' flag flipped from 0 to 1
    original_relevance_of_successful = df.loc[successful_assignments.index, RELEVANT_COLUMN]
    flipped_relevance_count = (original_relevance_of_successful == 0).sum()

    # Use the index of successful assignments to update the main DataFrame
    # 1. Assign the new DEMAND_ID
    df.loc[successful_assignments.index, LABEL_COLUMN] = successful_assignments['predicted_label']
    # 2. Correct the RELEVANT flag to 1
    df.loc[successful_assignments.index, RELEVANT_COLUMN] = 1

    print(f"✅ Corrected 'RELEVANT' flag from 0 to 1 for {flipped_relevance_count} rows.")
else:
    print("\n⚠️ No sentences met the high-confidence threshold. No changes were made.")

# --- VERIFICATION (AFTER) ---
print("\n--- DATA STATE (AFTER SCRIPT) ---")
print("Value counts for 'RELEVANT' column (this has been updated):")
print(df[RELEVANT_COLUMN].value_counts())
print(f"\nNumber of rows with an empty '{LABEL_COLUMN}': {df[LABEL_COLUMN].isnull().sum()}")
# ---

# Step 7: Save the final, corrected result
df.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"\n✅ All done! The updated dataset has been saved to: '{OUTPUT_FILE_PATH}'")
