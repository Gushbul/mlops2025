# Step 3: Basic ML Pipeline

Complete ML pipeline with preprocessing, training, and evaluation.

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

Pipeline stages:
1. `get_data` - downloads tips dataset
2. `preprocess` - creates high_tip feature
3. `train` - trains LogisticRegression
4. `evaluate` - prints accuracy to stdout
