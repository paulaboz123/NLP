from azure.ai.language.conversations import ConversationAuthoringClient
from azure.core.credentials import AzureKeyCredential
import yaml

def train_clu(endpoint, api_key, project_name):
    """
    Train a CLU project using Azure Language Studio.
    """
    client = ConversationAuthoringClient(endpoint, AzureKeyCredential(api_key))

    print(f"Starting CLU training for project: {project_name}...")
    poller = client.begin_train_project(project_name, training_mode="standard")
    result = poller.result()

    if result.status == "succeeded":
        print(f"CLU training succeeded for project: {project_name}")
    else:
        print(f"CLU training failed: {result.error}")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_clu(
        endpoint=config["azure"]["endpoint"],
        api_key=config["azure"]["api_key"],
        project_name=config["clu_project"]["project_name"]
    )