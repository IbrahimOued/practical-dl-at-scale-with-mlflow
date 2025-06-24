import click
import mlflow
import os
import zipfile
import requests
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@click.command(help="This program downloads data for finetuning a deep learning model for sentimental classification.")
@click.option("--download_url", default="https://pl-flash-data.s3.amazonaws.com/imdb.zip", help="This is the remote url where the data will be downloaded")
@click.option("--local_folder", default="./data", help="This is a local data folder.")
@click.option("--pipeline_run_name", default="chapter04", help="This is a mlflow run name.")
def task(download_url, local_folder, pipeline_run_name):
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        logger.info("Downloading data from  %s", download_url)
        # =========== start downloading data ==============
        download_and_extract_zip(download_url, local_folder)
        # =========== end downloading data ==============
        mlflow.log_param("download_url", download_url)
        mlflow.log_param("local_folder", local_folder)
        mlflow.log_param("mlflow run id", mlrun.info.run_id)
        mlflow.set_tag('pipeline_step', __file__)
        mlflow.log_artifacts(local_folder, artifact_path="data")

    logger.info("finished downloading data to %s", local_folder)

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

if __name__ == '__main__':
    task()