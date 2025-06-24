import logging
import os

import click
import mlflow
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "download_dataset",
    "fine_tuning_model",
    "register_model"
]

# Load environment variables
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY


@click.command()
@click.option("--steps", default="all", type=str)
def run_pipeline(steps):


    EXPERIMENT_NAME = "dl_at_scale_model_chapter04"
    mlflow.set_tracking_uri("http://localhost") # important to run the experiment inside the docker experimentation env
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    logger.info("pipeline experiment_id: %s", experiment.experiment_id)

    # Steps to execute
    active_steps = steps.split(",") if steps != "all" else _steps
    logger.info("pipeline active steps to execute in this run: %s", active_steps)

    with mlflow.start_run(run_name='pipeline', nested=True) as active_run:
        if "download_dataset" in active_steps:
            download_run = mlflow.run(".", "download_dataset", parameters={})
            download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)
            file_path_uri = download_run.data.params['local_folder']
            logger.info('downloaded data is located locally in folder: %s', file_path_uri)
            logger.info(download_run)

        if "create_dataset" in active_steps:
            create_dataset_run = mlflow.run(".", "create_dataset", parameters={"data_path": file_path_uri, "train_test_val_path": file_path_uri})
            create_dataset_run = mlflow.tracking.MlflowClient().get_run(create_dataset_run.run_id)
            logger.info(create_dataset_run)

        if "fine_tuning_model" in active_steps:
            fine_tuning_run = mlflow.run(".", "fine_tuning_model", parameters={"data_path": file_path_uri})
            fine_tuning_run_id = fine_tuning_run.run_id
            fine_tuning_run = mlflow.tracking.MlflowClient().get_run(fine_tuning_run_id)
            logger.info(fine_tuning_run)

        if "register_model" in active_steps:
            if fine_tuning_run_id is not None and fine_tuning_run_id != 'None':
                register_model_run = mlflow.run(".", "register_model", parameters={"mlflow_run_id": fine_tuning_run_id})
                register_model_run = mlflow.tracking.MlflowClient().get_run(register_model_run.run_id)
                logger.info(register_model_run)
            else:
                logger.info("no model to register since no trained model run id.")
    logger.info('finished mlflow pipeline run with a run_id = %s', active_run.info.run_id)


if __name__ == "__main__":
    run_pipeline()