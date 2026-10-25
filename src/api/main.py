# Goal: Create a FastAPI app to serve your trained ML model into a web service that anyone 
# (or any system) can call over HTTP.

from fastapi import FastAPI            # Web framework for APIs
from pathlib import Path               # For handling file paths cleanly
from typing import List, Dict, Any     # For type hints (clarity in endpoints)
import pandas as pd                    # To handle incoming JSON as DataFrames

# Import inference pipeline
from src.inference_pipeline.inference import predict
from src.artifacts import ensure_artifact

# ----------------------------
# Config
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------
# Paths
# ----------------------------
# Serving artifacts are read locally first and downloaded from Hugging Face
# when the container does not already contain them.
MODEL_PATH = ensure_artifact("models/xgb_best_model.pkl", PROJECT_ROOT)
TRAIN_FE_PATH = ensure_artifact(
    "data/processed/feature_engineered_train.csv", PROJECT_ROOT
)
FREQ_ENCODER_PATH = ensure_artifact("models/freq_encoder.pkl", PROJECT_ROOT)
TARGET_ENCODER_PATH = ensure_artifact("models/target_encoder.pkl", PROJECT_ROOT)

# Load expected training features for alignment
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"]
else:
    TRAIN_FEATURE_COLUMNS = None

# ----------------------------
# App
# ----------------------------
# Instantiates the FastAPI app.
app = FastAPI(title="Housing Regression API")

# / → simple landing endpoint to confirm API is alive.
@app.get("/")
def root():
    return {"message": "Housing Regression API is running 🚀"}

# /health → checks if model exists, returns status info (like expected feature count).
@app.get("/health")
def health():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}
    if not MODEL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model not found"
    else:
        status["status"] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)
    return status

# Prediction Endpoint: This is the core ML serving endpoint.
@app.post("/predict")
def predict_batch(data: List[dict]):
    if not MODEL_PATH.exists():
        return {"error": f"Model not found at {str(MODEL_PATH)}"}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}

    preds_df = predict(
        df,
        model_path=MODEL_PATH,
        freq_encoder_path=FREQ_ENCODER_PATH,
        target_encoder_path=TARGET_ENCODER_PATH,
    )

    resp = {"predictions": preds_df["predicted_price"].astype(float).tolist()}
    if "actual_price" in preds_df.columns:
        resp["actuals"] = preds_df["actual_price"].astype(float).tolist()

    return resp

# Batch runner
from src.batch.run_monthly import run_monthly_predictions

# Trigger a monthly batch job via API.
@app.post("/run_batch")
def run_batch():
    preds = run_monthly_predictions()
    return {
        "status": "success",
        "rows_predicted": int(len(preds)),
        "output_dir": "data/predictions/"
    }

# Returns a preview of the most recent batch predictions.
@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    pred_dir = Path("data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))
    if not files:
        return {"error": "No predictions found"}

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }


"""
🔹 Execution Order / Module Flow

1. Imports (FastAPI, pandas, and the inference function).
2. Resolve serving artifacts packaged in the image.
3. Infer the expected feature schema when the training feature file is present.
4. Create FastAPI app (app = FastAPI).
5. Declare endpoints (/, /health, /predict, /run_batch, /latest_predictions).
"""