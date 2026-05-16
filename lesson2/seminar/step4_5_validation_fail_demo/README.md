# Step 4.5: Data Validation Fail Fast

ML pipeline with intentionally bad data to demonstrate validation failure and fail-fast behavior.

## Setup
```bash
make install
dvc init
dvc remote add -d local ../../.dvcstore
```

## Run
```bash
dvc repro
```

Pipeline will **fail at validation step** because get_data creates dataset with violations:
- `total_bill`: contains null values and value > 100
- `tip`: contains null values
- `size`: contains values < 1 and > 10

This demonstrates **fail fast** principle: data quality issues are caught early, before expensive ML training stages execute.

Check validation report at `reports/validation/index.html` for specific violation details.
