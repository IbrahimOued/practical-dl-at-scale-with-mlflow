import torch
import flash
import mlflow
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier

download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")

datamodule = TextClassificationData.from_csv(
    "review",
    "sentiment",
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv",
    batch_size=4,
)

classifier_model = TextClassifier(backbone="prajjwal1/bert-tiny", num_classes=datamodule.num_classes)
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(classifier_model, datamodule=datamodule, strategy="freeze")

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="chapter02"):
    trainer.finetune(classifier_model, datamodule=datamodule, strategy="freeze")
    trainer.test()