from azure.ai.language.conversations import ConversationAuthoringClient, ConversationAnalysisClient
from azure.core.credentials import AzureKeyCredential
import time

# Azure resource details
endpoint = "https://<your-resource-name>.cognitiveservices.azure.com/"
key = "<your-key>"

# Existing project details
project_name = "existing_project_name"  # Replace with your project's name
deployment_name = "production"  # Replace with your deployment name (e.g., "production" or "staging")

# Initialize the clients
authoring_client = ConversationAuthoringClient(endpoint, AzureKeyCredential(key))
analysis_client = ConversationAnalysisClient(endpoint, AzureKeyCredential(key))

# Retrieve training data from the existing project
training_data = authoring_client.get_training_data(project_name)
print("Training Data:")
for utterance in training_data.utterances:
    print(f"Text: {utterance.text}")
    if hasattr(utterance, 'intent'):
        print(f"  Intent: {utterance.intent}")
    print(f"  Entities: {[(e.category, e.offset, e.length) for e in utterance.entities]}")

# Optional: Add new training data
new_training_data = {
    "utterances": [
        {
            "text": "Can I book a flight to Paris?",
            "intent": "BookFlight",  # Include intent for CLU, omit for CNER
            "entities": [
                {"category": "Location", "offset": 23, "length": 5}
            ]
        }
    ]
}
authoring_client.add_training_data(project_name, new_training_data)
print("New training data added successfully!")

# Trigger training for the existing project
train_response = authoring_client.train_model(project_name)
print(f"Training initiated. Status: {train_response.status}")

# Optional: Check training status
while train_response.status.lower() not in ("succeeded", "failed"):
    print(f"Training status: {train_response.status}")
    time.sleep(10)  # Wait for 10 seconds before checking again
    train_response = authoring_client.get_project(project_name)

# Test the deployed model with a query
query = "Book me a flight to New York tomorrow."
response = analysis_client.analyze_conversation(
    project_name=project_name,
    deployment_name=deployment_name,
    query=query
)

# Print the analysis results
if hasattr(response, "top_intent"):
    print(f"Top Intent: {response.top_intent}")  # For CLU
if hasattr(response, "entities"):
    print("Entities:")
    for entity in response.entities:
        print(f"  - {entity.category}: {entity.text}")
