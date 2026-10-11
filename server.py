"""
server.py — API ligera para servir predicciones con modelo ONNX.

Endpoints:
    GET  /          — serving static index.html
    GET  /static/*  — archivos estaticos (CSS, JS)
    POST /predict   — prediccion con modelo ONNX
    GET  /health    — health check

Uso:
    uvicorn server:app --host 0.0.0.0 --port 8000
    # o directamente:
    python server.py
"""

from __future__ import annotations
import os, sys, warnings, numpy as np
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Model loading ──────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "models")
MODEL_FILE = None
ONNX_RT = None
model_session = None

def load_onnx_model():
    global MODEL_FILE, ONNX_RT, model_session
    if ONNX_RT is not None:
        return model_session
    try:
        import onnxruntime as ort
        ONNX_RT = ort
    except ImportError:
        print("[WARN] onnxruntime no disponible. Modo mock activo.")
        return None
    model_dir = Path(MODEL_PATH)
    onnx_files = list(model_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"[WARN] No se encontro ningun modelo ONNX en {model_dir}")
        return None
    MODEL_FILE = str(onnx_files[0])
    print(f"[OK] Modelo ONNX cargado: {MODEL_FILE}")
    try:
        session = ort.InferenceSession(MODEL_FILE, providers=["CPUExecutionProvider"])
        print(f"  Proveedor: {session.get_providers()}")
        print(f"  Inputs: {[i.name for i in session.get_inputs()]}")
        print(f"  Outputs: {[o.name for o in session.get_outputs()]}")
        model_session = session
        return session
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        return None

# ── Pydantic models ───────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    home_team: str | None = None
    away_team: str | None = None
    league: str = "laliga"
    features: dict[str, Any] = Field(default_factory=dict)
    use_mock: bool = False

class PredictionOutput(BaseModel):
    home_team: str
    away_team: str
    prob_home: float
    prob_draw: float
    prob_away: float
    prob_btts: float
    ou: dict[str, float]
    xg_home: float
    xg_away: float
    confidence: str
    top_scores: list[dict] = Field(default_factory=list)

# ── Prediction logic ──────────────────────────────────────────────────────
def predict_onnx(session, features: dict) -> dict | None:
    if session is None:
        return None
    feature_keys = ["xg_home","xg_away","elo_home","elo_away",
                    "home_advantage","is_derby","xg_diff","elo_diff",
                    "strength_home","strength_away"]
    values = []
    for key in feature_keys:
        val = features.get(key)
        if val is None:
            defaults = {"xg_home":1.5,"xg_away":1.0,"elo_home":1800,
                        "elo_away":1750,"home_advantage":0.15,"is_derby":0}
            if key in ("xg_diff","elo_diff"):
                val = float(features.get("xg_home",1.5)) - float(features.get("xg_away",1.0)) if key=="xg_diff" else float(features.get("elo_home",1800)) - float(features.get("elo_away",1750))
            else:
                val = defaults.get(key, 0.0)
        values.append(float(val))
    X = np.array([values], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: X})
    probs = output[0].flatten()
    probs = probs / probs.sum()
    return {"prob_home": float(probs[0]), "prob_draw": float(probs[1]), "prob_away": float(probs[2])}

def predict_mock(home_team: str, away_team: str, features: dict) -> dict:
    home_adv = float(features.get("home_advantage", 0.15))
    xg_h = float(features.get("xg_home", 1.5))
    xg_a = float(features.get("xg_away", 1.0))
    xg_diff = xg_h - xg_a
    elo_diff = (float(features.get("elo_home", 1800)) - float(features.get("elo_away", 1750))) / 100.0
    raw_home = 0.40 + xg_diff * 0.15 + elo_diff * 0.10 + home_adv
    raw_draw = 0.28 - abs(xg_diff) * 0.05
    raw_away = 1.0 - raw_home - raw_draw
    total = max(raw_home + raw_draw + raw_away, 0.001)
    prob_home = max(0.05, min(0.75, raw_home / total))
    prob_draw = max(0.10, min(0.40, raw_draw / total))
    prob_away = max(0.05, 1.0 - prob_home - prob_draw)
    total = prob_home + prob_draw + prob_away
    prob_home, prob_draw, prob_away = prob_home/total, prob_draw/total, prob_away/total
    prob_btts = max(0.2, min(0.75, 1.0 - np.exp(-(xg_h + xg_a) * 0.8)))
    total_goals = xg_h + xg_a
    ou = {str(k): max(0.05, min(0.95, 1.0 - np.exp(-total_goals / (k + 0.5 + 0.5)))) for k in range(5)}
    max_prob = max(prob_home, prob_draw, prob_away)
    confidence = "ALTA" if max_prob > 0.55 else ("MEDIA" if max_prob > 0.40 else "BAJA")
    scores = []
    for h in range(4):
        for a in range(4):
            prob = (np.exp(-xg_h) * xg_h**h / np.math.factorial(h)) * (np.exp(-xg_a) * xg_a**a / np.math.factorial(a))
            if prob > 0.01:
                scores.append({"score": f"{h}-{a}", "prob": prob, "prob_pct": round(prob * 100, 1)})
    scores.sort(key=lambda x: -x["prob"])
    top_scores = scores[:6]
    return {
        "home_team": home_team or "Local", "away_team": away_team or "Visitante",
        "prob_home": round(prob_home, 4), "prob_draw": round(prob_draw, 4), "prob_away": round(prob_away, 4),
        "prob_btts": round(prob_btts, 4),
        "ou": {k: round(v, 4) for k, v in ou.items()},
        "xg_home": round(xg_h, 2), "xg_away": round(xg_a, 2),
        "confidence": confidence, "top_scores": top_scores,
    }

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(title="Edgebet Demo API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.on_event("startup")
async def startup():
    load_onnx_model()
    if model_session is None:
        print("[INFO] Ejecutando en modo mock (sin modelo ONNX)")
        print("[INFO] Para activar el modelo, pon un .onnx en models/")

@app.get("/")
async def root():
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return {"message": "Edgebet Demo API", "status": "running"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if input_data.use_mock:
        result = predict_mock(input_data.home_team or "Local", input_data.away_team or "Visitante", input_data.features or {})
        return PredictionOutput(**result)
    features = input_data.features or {}
    result = predict_onnx(model_session, features)
    if result is not None:
        mock = predict_mock(input_data.home_team or "Local", input_data.away_team or "Visitante", features)
        mock.prob_home = result["prob_home"]
        mock.prob_draw = result["prob_draw"]
        mock.prob_away = result["prob_away"]
        return PredictionOutput(**mock)
    return PredictionOutput(**predict_mock(input_data.home_team or "Local", input_data.away_team or "Visitante", features))

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_FILE if MODEL_FILE else "none (mock mode)", "onnx_available": ONNX_RT is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Edgebet Demo API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
