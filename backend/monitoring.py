import os
import pandas as pd
from github import Github
from evidently import Dataset, DataDefinition, Report, Regression
from evidently.presets import DataDriftPreset, RegressionPreset, DataSummaryPreset
from evidently.ui.workspace import CloudWorkspace
from pathlib import Path

EVIDENTLY_TOKEN = os.environ["EVIDENTLY_API_TOKEN"]
PROJECT_ID = os.environ["EVIDENTLY_PROJECT_ID"]
GITHUB_TOKEN = os.environ["GITHUB_PAT"]
REPO_NAME = os.environ["GITHUB_REPO"]

TARGET = "median_house_value"
PREDICTION_COL = "prediction"
REFERENCE_PATH = "reference_data.csv"  # path inside the repo

REFERENCE_PATH = Path(__file__).parent.parent / "reference_data.csv"

def _load_reference_from_github() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_PATH)
    
    # contents.decoded_content handles the base64 decoding for you
    import io
    csv_bytes = contents.decoded_content
    return pd.read_csv(io.BytesIO(csv_bytes))

def _build_schema(df: pd.DataFrame) -> DataDefinition:
    feature_cols = [c for c in df.columns if c not in [TARGET, PREDICTION_COL]]
    numerical = [c for c in feature_cols if df[c].dtype in ["float64", "int64"]]
    categorical = [c for c in feature_cols if c not in numerical]
    return DataDefinition(
        numerical_columns=numerical,
        categorical_columns=categorical if categorical else None,
        regression=[Regression(target=TARGET, prediction=PREDICTION_COL)],
    )

def run_monitoring(df: pd.DataFrame, batch_name: str):
    try:
        df_ref = _load_reference_from_github()
        if TARGET in df_ref.columns:
            df_ref = df_ref.rename(columns={"median_house_value": TARGET})

        schema = _build_schema(df_ref)

        ref_dataset = Dataset.from_pandas(df_ref, data_definition=schema)
        curr_dataset = Dataset.from_pandas(df, data_definition=schema)

        report = Report([
            DataDriftPreset(),
            RegressionPreset(),
            DataSummaryPreset(),
        ])

        ws = CloudWorkspace(token=EVIDENTLY_TOKEN, url="https://app.evidently.cloud")
        project = ws.get_project(PROJECT_ID)

        result = report.run(curr_dataset, ref_dataset)
        ws.add_run(PROJECT_ID, result, include_data=False)
        print(f"Evidently: uploaded report for {batch_name}")

    except Exception as e:
        # Don't crash the background task — log and move on
        print(f"Monitoring failed for {batch_name}: {e}")