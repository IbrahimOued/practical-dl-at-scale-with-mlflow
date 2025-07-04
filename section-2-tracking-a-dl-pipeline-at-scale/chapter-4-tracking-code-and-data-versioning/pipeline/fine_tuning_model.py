import click
import mlflow
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
import torchmetrics
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()




@click.command(help="This program finetunes a deep learning model for sentimental classification.")
@click.option("--foundation_model", default="prajjwal1/bert-tiny", help="This is the pretrained backbone foundation model")
@click.option("--data_path", default="data", help="This is the path to data.")
def task(foundation_model, data_path):

    # datamodule = TextClassificationData.from_csv(
    #     "review",
    #     "sentiment",
    #     train_file=f"{data_path}/imdb/train.csv",
    #     val_file=f"{data_path}/imdb/valid.csv",
    #     test_file=f"{data_path}/imdb/test.csv",
    #     batch_size=64,
    # )

    # classifier_model = TextClassifier(backbone=foundation_model, num_classes=datamodule.num_classes, metrics=torchmetrics.F1Score(datamodule.num_classes))
    # trainer = flash.Trainer(max_epochs=20, gpus=torch.cuda.device_count())

    # mlflow.pytorch.autolog()
    # with mlflow.start_run(run_name="chapter04") as dl_model_tracking_run:
    #     trainer.finetune(classifier_model, datamodule=datamodule, strategy=fine_tuning_strategy)
    #     trainer.test(classifier_model, datamodule=datamodule)

    #     # mlflow log additional hyper-parameters used in this training
    #     mlflow.log_params(classifier_model.hparams)

    #     run_id = dl_model_tracking_run.info.run_id
    #     logger.info("run_id: {}; lifecycle_stage: {}".format(run_id, mlflow.get_run(run_id).info.lifecycle_stage))
    #     mlflow.log_param("fine_tuning_mlflow_run_id", run_id)
    #     mlflow.set_tag('pipeline_step', __file__)
    train(foundation_model, data_path)



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
    
def train(foundation_model, data_path): 
    train_df = pd.read_csv(os.path.join(data_path, "imdb/train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "imdb/valid.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "imdb/test.csv"))

    # Determine number of classes from the sentiment column
    num_classes = train_df['sentiment'].nunique()
    logger.info(f"Number of classes: {num_classes}")

    # Initialize Tokenizer
    model_name = foundation_model # As used in Flash example
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
    logger.info(f"Using device: {device}")
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

        metrics = {
            'accuracy': torchmetrics.Accuracy(task='binary'),
            'precision': torchmetrics.Precision(task='binary'),
            'recall': torchmetrics.Recall(task='binary'),
            'f1': torchmetrics.F1Score(task='binary')
        }

        mlflow.log_params(params)
        logger.info(f"Run ID: {run.info.run_id}")

        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        logger.info("\nStarting training...")
        for epoch in range(num_epochs):
            logger.info(f"---⌛ Epoch {epoch+1}/{num_epochs} ---")
            train_loss = train_epoch(model, train_dataloader, optimizer, device)
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_epoch(model, val_dataloader, device, metrics)

            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
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

        
        run_id = run.info.run_id
        logger.info("run_id: {}; lifecycle_stage: {}".format(run_id, mlflow.get_run(run_id).info.lifecycle_stage))


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
    
    # Define metrics
    metrics_list = ['accuracy', 'precision', 'recall', 'f1']

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

if __name__ == '__main__':
    task()