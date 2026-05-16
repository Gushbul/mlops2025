# Step 2: DVC Data Management

Downloads tips dataset and tracks it with DVC.

## Setup
```bash
make install
dvc init
dvc remote add -d local ../../.dvcstore
```

## Run
```bash
dvc repro
dvc push
```

Creates `data/raw/tips.csv` with Seaborn tips dataset.
