import os
import base64
import pandas as pd
from github import Github
from pathlib import Path

GITHUB_TOKEN = os.environ["GITHUB_PAT"]
REPO_NAME = os.environ["GITHUB_REPO"] 
DATA_DIR = "production_batches"

def push_df_to_github(df: pd.DataFrame, batch_name: str) -> str:
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    path = f"{DATA_DIR}/{batch_name}.csv"

    try:
        # If file somehow exists, update it
        existing = repo.get_contents(path)
        repo.update_file(
            path=path,
            message=f"data: add batch {batch_name}",
            content=csv_bytes,
            sha=existing.sha,
        )
    except Exception:
        # File doesn't exist yet — create it
        repo.create_file(
            path=path,
            message=f"data: add batch {batch_name}",
            content=csv_bytes,
        )

    return path