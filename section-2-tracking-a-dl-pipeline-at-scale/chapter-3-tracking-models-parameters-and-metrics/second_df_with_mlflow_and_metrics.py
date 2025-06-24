import torch
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import os
import zipfile
import requests
from tqdm.auto import tqdm
import mlflow
import torchmetrics

from dotenv import load_dotenv
# Load environment variables
load_dotenv('./mlflow_docker_setup/.env')
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

# Set up MLflow experiment
EXPERIMENT_NAME = "dl_model_chapter2"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)
mlflow.pytorch.autolog()

# Define metrics
metrics_list = ['accuracy', 'precision', 'recall', 'f1']

# Initialize torchmetrics
metrics = {
    'accuracy': torchmetrics.Accuracy(task='binary'),
    'precision': torchmetrics.Precision(task='binary'),
    'recall': torchmetrics.Recall(task='binary'),
    'f1': torchmetrics.F1Score(task='binary')
}

# --- 1. Data Download Utility (similar to Flash's download_data) ---
def download_and_extract_zip(url, path):
    if not os.path.exists(path):
        os.makedirs(path)
    zip_file_name = os.path.join(path, "downloaded_data.zip")

    print(f"Downloading data from {url}...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(zip_file_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong during download")
        return

    print(f"Extracting {zip_file_name} to {path}...")
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(path)
    os.remove(zip_file_name) # Clean up the zip file
    print("Download and extraction complete.")

# --- 2. Custom Dataset for IMDB (replaces TextClassificationData) ---
class IMDBDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure 'sentiment' is mapped to numerical labels
        # 0 for negative, 1 for positive
        self.label_map = {'negative': 0, 'positive': 1}
        self.dataframe['sentiment_label'] = self.dataframe['sentiment'].map(self.label_map)


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        review = str(self.dataframe.loc[idx, 'review'])
        label = self.dataframe.loc[idx, 'sentiment_label']

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt', # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 3. Training Function ---
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

# --- 4. Evaluation Function ---
def evaluate_epoch(model, dataloader, device, metrics):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Update metrics
    for metric_name in metrics_list:
        metrics[metric_name](torch.tensor(all_predictions), torch.tensor(all_true_labels))

    # Compute metrics
    accuracy = metrics['accuracy'].compute().item()
    precision = metrics['precision'].compute().item()
    recall = metrics['recall'].compute().item()
    f1 = metrics['f1'].compute().item()

    # Reset metrics
    for metric_name in metrics_list:
        metrics[metric_name].reset()

    return avg_loss, accuracy, precision, recall, f1

# --- Main Script ---
if __name__ == "__main__":
    # --- 1. & 2. Data Download and Preparation ---
    data_url = "https://pl-flash-data.s3.amazonaws.com/imdb.zip"
    data_path = "./data/"
    download_and_extract_zip(data_url, data_path)

    train_df = pd.read_csv(os.path.join(data_path, "imdb/train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "imdb/valid.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "imdb/test.csv"))

    # Determine number of classes from the sentiment column
    num_classes = train_df['sentiment'].nunique()
    print(f"Number of classes: {num_classes}")

    # Initialize Tokenizer
    model_name = "prajjwal1/bert-tiny" # As used in Flash example
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_token_length = 128 # A common max length for BERT-tiny, adjust as needed

    # Create custom datasets
    train_dataset = IMDBDataset(train_df, tokenizer, max_token_length)
    val_dataset = IMDBDataset(val_df, tokenizer, max_token_length)
    test_dataset = IMDBDataset(test_df, tokenizer, max_token_length)

    # Create DataLoaders
    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # --- 3. Model Definition and Setup ---
    # Use AutoModelForSequenceClassification for classification tasks with Hugging Face models
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Optimizer and Learning Rate
    optimizer = AdamW(model.parameters(), lr=2e-5) # Common learning rate for fine-tuning BERT

    # --- 4. Training Loop ---
    num_epochs = 10 # As specified in PyTorch Lightning example

    with mlflow.start_run() as run:
        params = {
            "model_name": model_name,
            "num_epochs": num_epochs,
            "num_classes": num_classes,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "metric_function" : "accuracy",
            "device": str(device),
            "num_gpus": torch.cuda.device_count(),
            "batch_size": batch_size,
            "learning_rate": 2e-5,
            "max_token_length": max_token_length
        }
        mlflow.log_params(params)
        print(f"Run ID: {run.info.run_id}")

        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        print("\nStarting training...")
        for epoch in range(num_epochs):
            print(f"---⌛ Epoch {epoch+1}/{num_epochs} ---")
            train_loss = train_epoch(model, train_dataloader, optimizer, device)
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_epoch(model, val_dataloader, device, metrics)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
                    f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, artifact_path="model")

        # --- 5. Testing ---
        # Once the fine-tuning step is completed, we will test the accuracy
        # of the model by running trainer.test():
        print("\nStarting testing...")
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_epoch(model, test_dataloader, device, metrics)
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
                f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

        # Log test metrics to MLflow
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)

        # --- 6. Prediction Example ---
        print("\nExample Prediction on new data:")
        model.eval()
        new_reviews = [
            "This movie was absolutely fantastic! I loved every moment of it.",
            "Utterly disappointing. A complete waste of time.",
            "It was okay, nothing special.",
            "The acting was superb, but the plot was a bit weak."
        ]

        for review in new_reviews:
            encoding = tokenizer.encode_plus(
                review,
                add_special_tokens=True,
                max_length=max_token_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_label_id = torch.argmax(logits, dim=-1).item()
                predicted_sentiment = "positive" if predicted_label_id == 1 else "negative"
                print(f"Review: '{review}' -> Predicted Sentiment: {predicted_sentiment}")
