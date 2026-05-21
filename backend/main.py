import io
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from model import load_model, predict, compute_rmse
from schemas import PredictionResponse
from github_client import push_df_to_github
from monitoring import run_monitoring


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    print("Model loaded")
    yield


app = FastAPI(title="ML Monitoring API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def save_to_github(df: pd.DataFrame, batch_name: str):
    try:
        path = push_df_to_github(df, batch_name)
        print(f"GitHub: pushed {path}")
    except Exception as e:
        print(f"GitHub push failed for {batch_name}: {e}")


def run_monitoring_task(df: pd.DataFrame, batch_name: str):
    try:
        run_monitoring(df, batch_name)
    except Exception as e:
        print(f"Monitoring failed for {batch_name}: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")

    contents = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV")

    if "median_house_value" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail="CSV must contain a 'median_house_value' column",
        )

    preds = predict(app.state.model, df)

    df["prediction"] = preds
    rmse = compute_rmse(df["median_house_value"], preds)

    batch_name = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    background_tasks.add_task(save_to_github, df.copy(), batch_name)
    background_tasks.add_task(run_monitoring_task, df.copy(), batch_name)

    return PredictionResponse(
        predictions=preds.tolist(),
        rmse=rmse,
        batch_name=batch_name,
    )