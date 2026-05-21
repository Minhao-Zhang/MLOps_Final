import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path

MODEL_PATH = Path("models/champion_model.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict(model, df):
    feature_cols = [c for c in df.columns if c != "median_house_value"]
    X = df[feature_cols]
    preds = model.predict(X)
    return preds

def compute_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))