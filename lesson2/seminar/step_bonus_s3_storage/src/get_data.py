import os
import tempfile
import shutil
import pandas as pd
import yaml


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def download_data():
    params = load_params()
    url = params["urls"]["tips"]

    os.makedirs("data/raw", exist_ok=True)

    df = pd.read_csv(url)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        tmp_path = tmp_file.name

    shutil.move(tmp_path, "data/raw/tips.csv")


if __name__ == "__main__":
    download_data()
