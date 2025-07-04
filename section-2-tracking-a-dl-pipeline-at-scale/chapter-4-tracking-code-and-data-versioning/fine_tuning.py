# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%  Notebook for fine-tuning a pretrained language model to do text-based sentiment classification

# %%
import os
import flash
import mlflow
import torch
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier
import torchmetrics


# %%
download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")
datamodule = TextClassificationData.from_csv(
    "review",
    "sentiment",
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv",
    batch_size=16
)


# %%
classifier_model = TextClassifier(backbone="prajjwal1/bert-tiny", num_classes=datamodule.num_classes, metrics=torchmetrics.F1Score(datamodule.num_classes))
trainer = flash.Trainer(max_epochs=25, gpus=torch.cuda.device_count())


# %%
EXPERIMENT_NAME = "mlflow_dl_model_chapter_04"
mlflow.set_tracking_uri('http://localhost:5000') # important to run the experiment inside the docker experimentation env
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)
REGISTERED_MODEL_NAME = 'dl_finetuned_model'


# %%
mlflow.pytorch.autolog()
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="chapter04") as dl_model_tracking_run:
    trainer.finetune(classifier_model, datamodule=datamodule, strategy="freeze")
    trainer.test(classifier_model, datamodule=datamodule)

    # mlflow log additional hyper-parameters used in this training
    mlflow.log_params(classifier_model.hparams)


# %%
run_id = dl_model_tracking_run.info.run_id
print("run_id: {}; lifecycle_stage: {}".format(run_id, mlflow.get_run(run_id).info.lifecycle_stage))


# %%
# register the fine-tuned model
logged_model = f'runs:/{run_id}/model'
mlflow.register_model(logged_model, REGISTERED_MODEL_NAME)
# %%
