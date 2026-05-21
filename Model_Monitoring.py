import pandas as pd
import numpy as np
import os
from evidently import Dataset, DataDefinition, Report, Regression
from evidently.presets import DataDriftPreset, RegressionPreset, DataSummaryPreset
from evidently.ui.workspace import CloudWorkspace

# Configuration
EVIDENTLY_API_TOKEN = "sk_prod.019e4b5f-f348-7e8d-be48-6979dcdd062c.rbIq9VjSIxkCukL_v7-TYDbLEPgLweiMOTEY2pqoS5g6RXRNJV-OqeBSYBEKWLXGiqKb1cM3s9SMoLeH-tk5sVLyZsH8ifhlc4UyPYmdhRFFy8aZY_YYholEszxp469o"
PROJECT_ID = "019e4b61-1e24-7420-ae8f-6023a3e4133c"
TARGET = "median_house_value"
PREDICTION_COL = "prediction"

REFERENCE_PATH = "reference_data.csv"
BATCH_PATHS = {
    "batch1_clean": "production_batches/batch1_clean.csv",
    "batch2_corrupted": "production_batches/batch2_corrupted.csv",
    "batch3_mixed": "production_batches/batch3_mixed.csv",
}

# Connect to Evidently Cloud
if not EVIDENTLY_API_TOKEN:
    raise ValueError(
        "EVIDENTLY_API_TOKEN not set.\n"
        "Run this in your terminal first:\n"
        "export EVIDENTLY_API_TOKEN=your_token_here"
    )

ws = CloudWorkspace(token=EVIDENTLY_API_TOKEN, url="https://app.evidently.cloud")
project = ws.get_project(PROJECT_ID)
print(f"Connected to project: {project.name}")

# Load reference dataset
df_reference = pd.read_csv(REFERENCE_PATH)

# Rename target to match if needed
if "median_house_value" in df_reference.columns:
    df_reference = df_reference.rename(columns={"median_house_value": TARGET})

print(f"Reference shape: {df_reference.shape}")
print(f"Columns: {df_reference.columns.tolist()}")

# Define data schema for Evidently
# All feature columns (everything except target and prediction)
feature_cols = [c for c in df_reference.columns if c not in [TARGET, PREDICTION_COL]]
numerical_feats   = [c for c in feature_cols if df_reference[c].dtype in ["float64", "int64"]]
categorical_feats = [c for c in feature_cols if c not in numerical_feats]

schema = DataDefinition(
    numerical_columns=numerical_feats,
    categorical_columns=categorical_feats if categorical_feats else None,
    regression=[Regression(
        target=TARGET,
        prediction=PREDICTION_COL,
    )]
)

print(f"Numerical cols: {numerical_feats}")
print(f"Categorical cols: {categorical_feats}")
print(f"Target: {TARGET}")
print(f"Prediction col: {PREDICTION_COL}")

# Build refernce evidently Dataset object
ref_dataset = Dataset.from_pandas(df_reference, data_definition=schema)

os.makedirs("evidently_reports", exist_ok=True)
 
for batch_name, batch_path in BATCH_PATHS.items():
    print(f"\n{batch_name}")
 
    df_batch = pd.read_csv(batch_path)
 
    if "median_house_value" in df_batch.columns:
        df_batch = df_batch.rename(columns={"median_house_value": TARGET})
 
    curr_dataset = Dataset.from_pandas(df_batch, data_definition=schema)
 
    # Build and run report with 3 presets
    report = Report([
        DataDriftPreset(),
        RegressionPreset(),
        DataSummaryPreset(),
    ])
 
    my_eval = report.run(curr_dataset, ref_dataset)

    # Upload to Evidently Cloud
    ws.add_run(PROJECT_ID, my_eval, include_data=False)
    print(f"Uploaded to Evidently Cloud: {batch_name}")

    # Also save locally as HTML backup
    local_path = f"evidently_reports/{batch_name}_report.html"
    my_eval.save_html(local_path)
    print(f"Saved locally: {local_path}")