"""
AI Model Inference Server
FastAPI-based serving layer compatible with KServe V2 inference protocol
"""

import time
import logging
import numpy as np
from typing import List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from starlette.responses import Response
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Prometheus Metrics (safe registry — no duplicates on reload) ──────────────
def get_or_create_metric(metric_class, name, description, labels=None, **kwargs):
    """Get existing metric or create new one — prevents duplicate registration."""
    try:
        if labels:
            return metric_class(name, description, labels, **kwargs)
        return metric_class(name, description, **kwargs)
    except ValueError:
        # Already registered — return existing one from registry
        return REGISTRY._names_to_collectors.get(name)

REQUEST_COUNT = get_or_create_metric(
    Counter, "inference_requests_total", "Total inference requests", ["model", "status"]
)
REQUEST_LATENCY = get_or_create_metric(
    Histogram, "inference_request_duration_seconds", "Inference latency in seconds",
    ["model"], buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)
ACTIVE_REQUESTS = get_or_create_metric(
    Gauge, "inference_active_requests", "Currently active inference requests"
)
MODEL_LOADED = get_or_create_metric(
    Gauge, "model_loaded", "Whether model is loaded (1=yes 0=no)", ["model"]
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class InferenceInput(BaseModel):
    name: str
    shape: List[int]
    datatype: str = "FP32"
    data: List[Any]

class InferenceRequest(BaseModel):
    id: Optional[str] = None
    inputs: List[InferenceInput]

class InferenceOutput(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    data: List[Any]

class InferenceResponse(BaseModel):
    id: Optional[str]
    model_name: str
    model_version: str = "v1"
    outputs: List[InferenceOutput]

# ── Model ─────────────────────────────────────────────────────────────────────
class IrisClassifier:
    MODEL_NAME = "iris-classifier"
    MODEL_VERSION = "v1.0.0"
    CLASSES = ["setosa", "versicolor", "virginica"]

    def __init__(self):
        self.loaded = False
        self.load()

    def load(self):
        logger.info("Loading model weights...")
        time.sleep(0.3)
        self.weights = np.random.randn(4, 3)
        self.loaded = True
        if MODEL_LOADED:
            MODEL_LOADED.labels(model=self.MODEL_NAME).set(1)
        logger.info(f"Model '{self.MODEL_NAME}' loaded OK")

    def predict(self, inputs: np.ndarray) -> dict:
        logits = inputs @ self.weights
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        predicted_idx = np.argmax(probs, axis=1)
        return {
            "probabilities": probs.tolist(),
            "predicted_class": [self.CLASSES[i] for i in predicted_idx],
            "confidence": probs.max(axis=1).tolist()
        }

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Inference Server",
    description="KServe-compatible model inference API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

model = IrisClassifier()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model.loaded, "model": model.MODEL_NAME}

@app.get("/ready")
def ready():
    if not model.loaded:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}

@app.get("/v2/models/{model_name}")
def model_metadata(model_name: str):
    return {
        "name": model.MODEL_NAME,
        "versions": [model.MODEL_VERSION],
        "platform": "python",
        "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1, 4]}],
        "outputs": [{"name": "probabilities", "datatype": "FP32", "shape": [-1, 3]}],
    }

@app.post("/v2/models/{model_name}/infer", response_model=InferenceResponse)
async def infer(model_name: str, request: InferenceRequest):
    if not model.loaded:
        raise HTTPException(status_code=503, detail="Model not ready")

    if ACTIVE_REQUESTS:
        ACTIVE_REQUESTS.inc()
    start = time.time()

    try:
        raw = np.array(request.inputs[0].data, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        result = model.predict(raw)
        duration = time.time() - start

        if REQUEST_COUNT:
            REQUEST_COUNT.labels(model=model.MODEL_NAME, status="success").inc()
        if REQUEST_LATENCY:
            REQUEST_LATENCY.labels(model=model.MODEL_NAME).observe(duration)

        logger.info(f"Inference OK | class={result['predicted_class']} | latency={duration*1000:.1f}ms")

        return InferenceResponse(
            id=request.id,
            model_name=model.MODEL_NAME,
            model_version=model.MODEL_VERSION,
            outputs=[
                InferenceOutput(
                    name="probabilities",
                    shape=list(np.array(result["probabilities"]).shape),
                    datatype="FP32",
                    data=result["probabilities"]
                ),
                InferenceOutput(
                    name="predicted_class",
                    shape=[len(result["predicted_class"])],
                    datatype="BYTES",
                    data=result["predicted_class"]
                ),
            ]
        )
    except Exception as e:
        if REQUEST_COUNT:
            REQUEST_COUNT.labels(model=model.MODEL_NAME, status="error").inc()
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if ACTIVE_REQUESTS:
            ACTIVE_REQUESTS.dec()

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, workers=1)