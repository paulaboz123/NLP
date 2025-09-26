# ===============================================================
# Single-Block Script for SBERT Zero-Shot Data Labeling
# ===============================================================

# Step 0: Install necessary packages if you haven't already.
# This command can be run once in your terminal or notebook.
# !pip install -q pandas==2.2.2 torch==2.3.1 sentence-transformers==3.0.1

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import sys

# --- CONFIGURATION (ACTION REQUIRED) ---
# 1. Set the path to your data file.
FILE_PATH = "your_dataset.csv"

# 2. Set the names of the relevant columns in your file.
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
RELEVANT_COLUMN = "relevant"

# 3. Set a minimum confidence score (cosine similarity) to assign a label.
# A good starting point is between 0.4 and 0.6.
# Higher value = higher quality, but fewer labels assigned.
CONFIDENCE_THRESHOLD = 0.5

# 4. Set the name for the output file.
OUTPUT_FILE_PATH = "your_dataset_labeled.csv"
# --- END CONFIGURATION ---


print("--- Starting the SBERT Zero-Shot Labeling Process ---")

# Step 1: Load the dataset
try:
    df = pd.read_csv(FILE_PATH)
    print(f"✅ Successfully loaded dataset with {len(df)} rows from '{FILE_PATH}'.")
    df[RELEVANT_COLUMN] = df[RELEVANT_COLUMN].astype(int)
except FileNotFoundError:
    print(f"❌ ERROR: The file '{FILE_PATH}' was not found. Please check the file path and name.")
    sys.exit() # Stop the script if the file isn't found

# Step 2: Prepare the data for labeling
# Get the unique, existing labels from your data to use as candidates.
candidate_labels = df[LABEL_COLUMN].dropna().unique().tolist()

# Select the rows to be labeled: unlabeled ('label' is NaN) and relevant ('relevant' is 1).
rows_to_label_df = df[df[LABEL_COLUMN].isnull() & (df[RELEVANT_COLUMN] == 1)].copy()
texts_to_label = rows_to_label_df[TEXT_COLUMN].tolist()

if not texts_to_label:
    print("\n✅ Process finished: No relevant, unlabeled sentences were found to process.")
    sys.exit()
else:
    print(f"🔍 Found {len(candidate_labels)} unique candidate labels.")
    print(f"🔍 Found {len(texts_to_label)} relevant, unlabeled sentences to process.")

# Step 3: Load the SBERT model
# This will automatically use a GPU if available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'all-MiniLM-L6-v2'  # A fast and effective model
print(f"\n🔄 Loading SBERT model '{model_name}' onto device '{device}'...")
model = SentenceTransformer(model_name, device=device)
print("✅ Model loaded successfully.")

# Step 4: Generate embeddings for labels and sentences
print("\n🔄 Generating embeddings for labels...")
label_embeddings = model.encode(candidate_labels, convert_to_tensor=True, show_progress_bar=True)

print("\n🔄 Generating embeddings for sentences...")
sentence_embeddings = model.encode(texts_to_label, convert_to_tensor=True, show_progress_bar=True)

# Step 5: Calculate similarity and assign best labels based on the threshold
print("\n🔄 Calculating similarities and assigning labels...")
cosine_scores = util.cos_sim(sentence_embeddings, label_embeddings)

# Get the highest score (confidence) and the index of that score for each sentence
confidence_scores, label_indices = torch.max(cosine_scores, dim=1)

# Move results to CPU for easier processing
confidence_scores = confidence_scores.cpu().numpy()
label_indices = label_indices.cpu().numpy()

new_labels = []
labels_assigned_count = 0
# Iterate through each sentence's result
for score, index in zip(confidence_scores, label_indices):
    if score >= CONFIDENCE_THRESHOLD:
        # If score is high enough, assign the corresponding label
        new_labels.append(candidate_labels[index])
        labels_assigned_count += 1
    else:
        # Otherwise, the label remains None
        new_labels.append(None)

# Add the newly generated labels to the temporary dataframe
rows_to_label_df['new_label'] = new_labels

print(f"✅ Label assignment complete.")
print(f"📊 Assigned labels to {labels_assigned_count} of {len(texts_to_label)} sentences (based on a {CONFIDENCE_THRESHOLD} threshold).")

# Step 6: Update the main DataFrame and save the results
# Create a mapping from the DataFrame index to the new (non-null) label
new_labels_map = rows_to_label_df['new_label'].dropna()

# Use the map to fill in the missing values in the original DataFrame's label column
df[LABEL_COLUMN].fillna(new_labels_map, inplace=True)
print(f"\n🔄 Merging new labels into the main DataFrame...")

# Save the updated DataFrame to a new CSV file
df.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"\n✅ All done! The updated dataset has been saved to: '{OUTPUT_FILE_PATH}'")
