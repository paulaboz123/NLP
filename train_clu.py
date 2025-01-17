import requests
import yaml

# Azure resource details
ENDPOINT = "https://<your-resource-name>.cognitiveservices.azure.com/"
API_KEY = "<your-api-key>"

HEADERS = {
    "Ocp-Apim-Subscription-Key": API_KEY,
    "Content-Type": "application/json"
}

def load_yaml(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def upload_dataset(config):
    """Upload a dataset to the project."""
    project_name = config['project_name']
    dataset_name = config['dataset_name']
    dataset_file = config['dataset_file']
    
    with open(dataset_file, 'r') as file:
        dataset_content = file.read()
    
    url = f"{ENDPOINT}/language/authoring/v1.0/projects/{project_name}/datasets"
    payload = {
        "name": dataset_name,
        "format": config['dataset_format'],
        "content": dataset_content
    }
    
    response = requests.post(url, headers=HEADERS, json=payload)
    if response.status_code == 201:
        print(f"Dataset '{dataset_name}' uploaded successfully.")
    else:
        print(f"Error uploading dataset: {response.json()}")

def train_model(config):
    """Train a model for the project."""
    project_name = config['project_name']
    
    url = f"{ENDPOINT}/language/authoring/v1.0/projects/{project_name}/train"
    response = requests.post(url, headers=HEADERS)
    if response.status_code == 202:
        print("Training started successfully.")
    else:
        print(f"Error starting training: {response.json()}")

def deploy_model(config):
    """Deploy the trained model."""
    project_name = config['project_name']
    deployment_name = config['deployment_name']
    model_label = config['model_label']
    
    url = f"{ENDPOINT}/language/authoring/v1.0/projects/{project_name}/deployments/{deployment_name}"
    payload = {
        "trainedModelLabel": model_label
    }
    
    response = requests.put(url, headers=HEADERS, json=payload)
    if response.status_code == 201:
        print(f"Deployment '{deployment_name}' created successfully.")
    else:
        print(f"Error deploying model: {response.json()}")

# Load configurations
dataset_config = load_yaml("dataset.yml")
train_config = load_yaml("train.yml")
deploy_config = load_yaml("deploy.yml")

# Run operations
upload_dataset(dataset_config)
train_model(train_config)
deploy_model(deploy_config)



#########
"""
Dataset Configuration (dataset.yml)


project_name: "my-clu-project"
dataset_name: "training-dataset"
dataset_file: "path/to/your-dataset.json"
dataset_format: "json"
Training Configuration (train.yml)


project_name: "my-clu-project"
deployment_name: "production"
model_label: "v1"
Deployment Configuration (deploy.yml)


project_name: "my-clu-project"
deployment_name: "production"
model_label: "v1"

3. Workflow Description
Define YAML Files: Store project and dataset details in dataset.yml, train.yml, and deploy.yml.
Run Python Script: The script reads YAML configurations, uploads the dataset, triggers training, and deploys the model.
Dataset Validation: Ensure your dataset follows the required JSON format.
4. Considerations
No Terminal Needed: Everything, including dataset upload, training, and deployment, is handled via code.
Error Handling: Include detailed error handling for API responses (e.g., 400, 401, 404).
Batch Script Automation: If necessary, use a lightweight automation tool to execute the Python script in production environments.
"""
###

"""

Here’s the full Python script to handle the process programmatically. This includes reading labeled data from Language Studio, exporting it, training a model, and deploying the model without using the UI.

Script for End-to-End Workflow
This script:

Exports labeled data from a Language Studio project.
Downloads the dataset programmatically.
Triggers training of the model using the REST API.
Deploys the trained model to an endpoint.
"""

import requests
import yaml
import time

# Azure resource details
ENDPOINT = "https://<your-resource-name>.cognitiveservices.azure.com/"
API_KEY = "<your-api-key>"

HEADERS = {
    "Ocp-Apim-Subscription-Key": API_KEY,
    "Content-Type": "application/json"
}

def load_yaml(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def export_dataset(config):
    """Export labeled dataset from Language Studio."""
    project_name = config['project_name']
    dataset_name = config['dataset_name']

    # Trigger dataset export
    export_url = f"{ENDPOINT}/language/authoring/v1.0/projects/{project_name}/datasets/{dataset_name}/export"
    response = requests.post(export_url, headers=HEADERS)

    if response.status_code == 202:
        job_id = response.json()['jobId']
        print(f"Dataset export started. Job ID: {job_id}")

        # Monitor export status
        status_url = f"{ENDPOINT}/language/authoring/v1.0/projects/{project_name}/datasets/{dataset_name}/export/jobs/{job_id}"
        while True:
            status_response = requests.get(status_url, headers=HEADERS)
            status = status_response.json()
            if status["status"] == "succeeded":
                download_url = status["resultUrl"]
                print(f"Dataset exported successfully. Download URL: {download_url}")
                return download_url
            elif status["status"] == "failed":
                print("Dataset export faile

"""
Supporting YAML Files
dataset.yml

project_name: "my-clu-project"
dataset_name: "training-dataset"
dataset_format: "json"
train.yml

project_name: "my-clu-project"
deploy.yml

project_name: "my-clu-project"
deployment_name: "production"
model_label: "v1"
"""

"""
Explanation of Steps
Dataset Export:
Uses the export API to retrieve a downloadable link for the labeled dataset in Language Studio.
Dataset Download:
Downloads the dataset to a local file (exported_dataset.json).
Model Training:
Initiates the training process for the project using the REST API.
Model Deployment:
Deploys the trained model to an endpoint.
"""

import requests
import time

# Azure resource details
ENDPOINT = "https://<your-resource-name>.cognitiveservices.azure.com/"
API_KEY = "<your-api-key>"

HEADERS = {
    "Ocp-Apim-Subscription-Key": API_KEY,
    "Content-Type": "application/json"
}

# Project details
PROJECT_NAME = "my-clu-project"
DATASET_NAME = "training-dataset"
DEPLOYMENT_NAME = "production"
MODEL_LABEL = "v1"

def export_dataset(project_name, dataset_name):
    """Export labeled dataset from Language Studio."""
    export_url = f"{ENDPOINT}/language/authoring/v1.0/projects/{project_name}/datasets/{dataset_name}/export"
    response = requests.post(export_url, headers=HEADERS)

    if response.status_code == 202:
        job_id = response.json()['jobId']
        print(f"Dataset export started. Job ID: {job_id}")

        # Monitor export status
        status_url = f"{ENDPOINT}/language/authoring/v1.0/projects/{project_name}/datasets/{dataset_name}/export/jobs/{job_id}"
        while True:
            status_response = requests.get(status_url, headers=HEADERS)
            status = status_response.json()
            if status["status"] == "succeeded":
                download_url = status["resultUrl"]
                print(f"Dataset exported successfully. Download URL: {download_url}")
                return download_url
            elif status["status"] == "failed":
                print("Dataset export failed.")
                break
            else:
                print("Export in progress...")
                time.sleep(5)
    else:
        print(f"Error exporting dataset: {response.json()}")
        return None

def download_dataset(download_url, output_path):
    """Download the exported dataset."""
    response = requests.get(download_url)
    if response.status_code == 200:
        with open(output_path, "wb") as file:
            file.write(response.content)
        print(f"Dataset downloaded successfully to {output_path}")
    else:
        print(f"Error downloading dataset: {response.status_code}")

def train_model(project_name):
    """Train a model for the project."""
    url = f"{ENDPOINT}/language/authoring/v1.0/projects/{project_name}/train"
    response = requests.post(url, headers=HEADERS)
    if response.status_code == 202:
        print("Training started successfully.")
    else:
        print(f"Error starting training: {response.json()}")

def deploy_model(project_name, deployment_name, model_label):
    """Deploy the trained model."""
    url = f"{ENDPOINT}/language/authoring/v1.0/projects/{project_name}/deployments/{deployment_name}"
    payload = {
        "trainedModelLabel": model_label
    }
    
    response = requests.put(url, headers=HEADERS, json=payload)
    if response.status_code == 201:
        print(f"Deployment '{deployment_name}' created successfully.")
    else:
        print(f"Error deploying model: {response.json()}")

def main():
    """Main function to automate the entire workflow."""
    # Step 1: Export dataset
    download_url = export_dataset(PROJECT_NAME, DATASET_NAME)
    if download_url:
        # Step 2: Download dataset
        output_path = "exported_dataset.json"
        download_dataset(download_url, output_path)

        # Step 3: Train model
        train_model(PROJECT_NAME)

        # Step 4: Deploy model
        deploy_model(PROJECT_NAME, DEPLOYMENT_NAME, MODEL_LABEL)

if __name__ == "__main__":
    main()
