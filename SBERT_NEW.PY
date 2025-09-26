# ===============================================================
# SCRIPT: Fill in DEMAND_ID for Relevant, Unlabeled Sentences
# ===============================================================

# Step 0: Install necessary packages.
# You only need to run this cell once.
# !pip install -q pandas==2.2.2 torch==2.3.1 sentence-transformers==3.0.1

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import sys

# --- CONFIGURATION (ACTION REQUIRED) ---
# 1. Set the path to your data file.
FILE_PATH = "your_dataset.csv"  # <--- CHANGE THIS TO YOUR FILE PATH

# 2. Confirm your column names. I have used the ones you provided.
TEXT_COLUMN = "TEXT"
LABEL_COLUMN = "DEMAND_ID"
RELEVANT_COLUMN = "RELEVANT"

# 3. Set a confidence threshold (0.0 to 1.0) for assigning a label.
CONFIDENCE_THRESHOLD = 0.5

# 4. Set the name for the output file.
OUTPUT_FILE_PATH = "your_dataset_labeled.csv"
# --- END CONFIGURATION ---


print("--- Starting the Labeling Process ---")

# Step 1: Load your dataset
try:
    df = pd.read_csv(FILE_PATH)
    print(f"✅ Successfully loaded dataset with {len(df)} rows from '{FILE_PATH}'.")
    df[RELEVANT_COLUMN] = df[RELEVANT_COLUMN].astype(int)
except FileNotFoundError:
    print(f"❌ ERROR: The file '{FILE_PATH}' was not found. Please stop the script and check the path.")
    sys.exit()

# --- VERIFICATION (BEFORE) ---
print("\n--- DATA STATE (BEFORE SCRIPT) ---")
print("Value counts for the 'RELEVANT' column (this will not change):")
print(df[RELEVANT_COLUMN].value_counts())
print(f"\nNumber of rows with an empty '{LABEL_COLUMN}': {df[LABEL_COLUMN].isnull().sum()}")
# ---

# Step 2: Prepare the data for labeling
candidate_labels = df[LABEL_COLUMN].dropna().unique().tolist()
rows_to_label_df = df[df[LABEL_COLUMN].isnull() & (df[RELEVANT_COLUMN] == 1)].copy()
texts_to_label = rows_to_label_df[TEXT_COLUMN].astype(str).tolist() # Ensure text is string

if not texts_to_label:
    print("\n✅ Process finished: No relevant, unlabeled sentences were found to process.")
    sys.exit()
else:
    print(f"\n🔍 Found {len(candidate_labels)} unique candidate labels.")
    print(f"🔍 Found {len(texts_to_label)} relevant, unlabeled sentences to process.")

# Step 3: Load the SBERT model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'all-MiniLM-L6-v2'
print(f"\n🔄 Loading SBERT model '{model_name}' onto device '{device}'...")
model = SentenceTransformer(model_name, device=device)
print("✅ Model loaded successfully.")

# Step 4: Generate embeddings
print("\n🔄 Generating embeddings for labels...")
label_embeddings = model.encode(candidate_labels, convert_to_tensor=True, show_progress_bar=True)
print("\n🔄 Generating embeddings for sentences...")
sentence_embeddings = model.encode(texts_to_label, convert_to_tensor=True, show_progress_bar=True)

# Step 5: Calculate similarities and assign labels based on the threshold
print("\n🔄 Calculating similarities and assigning new labels...")
cosine_scores = util.cos_sim(sentence_embeddings, label_embeddings)
confidence_scores, label_indices = torch.max(cosine_scores, dim=1)
confidence_scores = confidence_scores.cpu().numpy()
label_indices = label_indices.cpu().numpy()

new_labels = []
labels_assigned_count = 0
for score, index in zip(confidence_scores, label_indices):
    if score >= CONFIDENCE_THRESHOLD:
        new_labels.append(candidate_labels[index])
        labels_assigned_count += 1
    else:
        new_labels.append(None) # Leave as None if confidence is too low

rows_to_label_df['new_label'] = new_labels
print(f"✅ Label assignment complete. Assigned {labels_assigned_count} new labels.")

# Step 6: Update the main DataFrame and save the results
print("\n🔄 Merging new labels back into the main DataFrame...")
new_labels_map = rows_to_label_df['new_label'].dropna()
df[LABEL_COLUMN].fillna(new_labels_map, inplace=True)

# --- VERIFICATION (AFTER) ---
print("\n--- DATA STATE (AFTER SCRIPT) ---")
print("Value counts for the 'RELEVANT' column (as expected, this has not changed):")
print(df[RELEVANT_COLUMN].value_counts())
print(f"\nNumber of rows with an empty '{LABEL_COLUMN}': {df[LABEL_COLUMN].isnull().sum()}")
# ---

# Step 7: Save the final result
df.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"\n✅ All done! The updated dataset has been saved to: '{OUTPUT_FILE_PATH}'")
