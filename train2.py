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


############
import requests
import json
import time

# Replace with your Azure Language Service details
endpoint = "https://<your-resource-name>.cognitiveservices.azure.com"
subscription_key = "<your-api-key>"
project_name = "<your-project-name>"

# URL for training the model
train_url = f"{endpoint}/language/authoring/projects/{project_name}/train"

# Headers for authentication and content type
headers = {
    "Ocp-Apim-Subscription-Key": subscription_key,
    "Content-Type": "application/json",
}

# Trigger training
def train_model():
    response = requests.post(train_url, headers=headers)
    if response.status_code == 202:
        print("Training started successfully!")
        operation_location = response.headers.get("Operation-Location")
        print(f"Training status URL: {operation_location}")
        return operation_location
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

# Check training status
def check_training_status(status_url):
    while True:
        response = requests.get(status_url, headers=headers)
        if response.status_code == 200:
            status = response.json()
            print("Training Status:", status["status"])
            if status["status"] in ["succeeded", "failed"]:
                return status
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return None
        time.sleep(10)  # Wait before checking again

# Main process
status_url = train_model()
if status_url:
    result = check_training_status(status_url)
    print("Training Completed. Final Status:", result)

