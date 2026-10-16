import asyncio
import json
import logging
import os
import queue
import threading
from typing import Any, Dict, List

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from river import compose, feature_extraction, linear_model, metrics, preprocessing, stream
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Model identifier on HuggingFace Hub. This model supports multilingual emotion
# classification. Adjust as needed.
HF_MODEL_NAME = os.getenv(
    "HF_MODEL_NAME",
    "cardiffnlp/twitter-xlm-roberta-base-emotion",
)

# Device selection: use GPU if available, otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("real_time_emotion_analyzer")

# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


class AnalyzeRequest(BaseModel):
    """Request payload for emotion analysis."""

    text: str = Field(..., min_length=1, description="Text to be analysed for emotion.")
    language: str = Field(
        "auto",
        description="ISO 639-1 language code or 'auto' for automatic detection.",
    )

    @validator("language")
    def validate_language(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("language must be a string")
        return v.lower()


class AnalyzeResponse(BaseModel):
    """Response payload containing the top emotion prediction."""

    text: str
    language: str
    top_emotion: str
    confidence: float
    all_predictions: List[Dict[str, Any]]


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_transformer_model(model_name: str):
    """Load a HuggingFace transformer model and tokenizer for sequence classification."""
    logger.info("Loading tokenizer for model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading model %s on device %s", model_name, DEVICE)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    # Use the HuggingFace pipeline for convenience.
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE.type == "cuda" else -1,
        return_all_scores=True,
    )
    logger.info("Model %s loaded successfully", model_name)
    return classifier


# --------------------------------------------------------------------------- #
# River online learner
# --------------------------------------------------------------------------- #

class OnlineEmotionLearner:
    """
    Simple online learner using River. It receives (text, label) pairs and updates
    a multinomial Naive Bayes classifier. The learner is deliberately lightweight
    to keep the real‑time footprint low.
    """

    def __init__(self):
        # Text vectorizer (bag‑of‑words) → Naive Bayes → Logistic regression for calibration.
        self.model = compose.Pipeline(
            feature_extraction.BagOfWords(),
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(),
        )
        self.metric = metrics.Accuracy()
        self.lock = threading.Lock()
        logger.info("Initialized online River learner")

    def learn_one(self, text: str, label: str) -> None:
        """Update the model with a single observation."""
        with self.lock:
            x = {"text": text}
            y = label
            self.model.learn_one(x, y)
            self.metric.update(y, self.model.predict_one(x))

    def predict_proba_one(self, text: str) -> Dict[str, float]:
        """Return probability distribution over known classes."""
        with self.lock:
            x = {"text": text}
            # River's predict_proba_one returns a dict {label: prob}
            return self.model.predict_proba_one(x)


# --------------------------------------------------------------------------- #
# FastAPI app and endpoints
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Real‑Time Multilingual Emotion Stream Analyzer",
    description="Analyse emotions from a live text stream using a multilingual transformer model "
    "and an online River learner for continuous adaptation.",
    version="1.0.0",
)

# Global objects (initialized lazily on first request to avoid heavy startup cost in Docker)
_transformer_classifier: pipeline = None
_online_learner: OnlineEmotionLearner = None
_learning_queue: queue.Queue = queue.Queue()


def get_transformer_classifier() -> pipeline:
    global _transformer_classifier
    if _transformer_classifier is None:
        _transformer_classifier = load_transformer_model(HF_MODEL_NAME)
    return _transformer_classifier


def get_online_learner() -> OnlineEmotionLearner:
    global _online_learner
    if _online_learner is None:
        _online_learner = OnlineEmotionLearner()
    return _online_learner


# --------------------------------------------------------------------------- #
# Background worker for online learning
# --------------------------------------------------------------------------- #

def learning_worker():
    """Continuously consume (text, label) tuples from the queue and update the River model."""
    learner = get_online_learner()
    while True:
        try:
            text, label = _learning_queue.get(timeout=1.0)
            logger.debug("Online learner received: %s -> %s", text[:30], label)
            learner.learn_one(text, label)
            _learning_queue.task_done()
        except queue.Empty:
            continue
        except Exception as exc:
            logger.exception("Error in online learning worker: %s", exc)


_thread = threading.Thread(target=learning_worker, daemon=True)
_thread.start()


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def format_predictions(raw_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw HuggingFace scores into a sorted list of emotion predictions."""
    # raw_scores is a list of dicts: [{'label': 'joy', 'score': 0.73}, ...]
    sorted_scores = sorted(raw_scores, key=lambda d: d["score"], reverse=True)
    return [{"emotion": s["label"], "confidence": round(s["score"], 4)} for s in sorted_scores]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Analyse a single piece of text and return the most probable emotion.
    The request can optionally include a language code; the underlying model
    handles multilingual input automatically.
    """
    classifier = get_transformer_classifier()
    try:
        # The pipeline returns a list of lists because return_all_scores=True.
        raw = classifier(request.text)[0]  # type: ignore[index]
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    predictions = format_predictions(raw)
    top = predictions[0]

    # Queue the observation for online learning if the client supplies a label via header.
    label = request.headers.get("X-Emotion-Label") if isinstance(request, Request) else None
    if label:
        _learning_queue.put_nowait((request.text, label))

    response = AnalyzeResponse(
        text=request.text,
        language=request.language,
        top_emotion=top["emotion"],
        confidence=top["confidence"],
        all_predictions=predictions,
    )
    return response


@app.get("/health")
async def health_check():
    """Simple health endpoint used by orchestration tools."""
    return JSONResponse(content={"status": "ok"})


@app.get("/metrics")
async def metrics_endpoint():
    """
    Expose basic online learning metrics (accuracy) for monitoring.
    """
    learner = get_online_learner()
    acc = learner.metric.get()
    return JSONResponse(content={"online_accuracy": round(acc, 4) if acc is not None else None})


# --------------------------------------------------------------------------- #
# Streaming endpoint (Server‑Sent Events)
# --------------------------------------------------------------------------- #

async def event_generator(texts: List[str]):
    """
    Async generator yielding emotion predictions for a list of texts.
    Each yielded line follows the Server‑Sent Events (SSE) format.
    """
    classifier = get_transformer_classifier()
    for txt in texts:
        raw = classifier(txt)[0]  # type: ignore[index]
        predictions = format_predictions(raw)
        top = predictions[0]
        payload = {
            "text": txt,
            "top_emotion": top["emotion"],
            "confidence": top["confidence"],
            "all_predictions": predictions,
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0)  # allow event loop to switch tasks


@app.post("/stream")
async def stream_emotions(request: Request):
    """
    Accept a JSON array of texts and stream back predictions using Server‑Sent Events.
    Example request body:
    ```json
    ["I am happy", "Je suis triste", "Ich bin wütend"]
    ```
    """
    try:
        body = await request.json()
        if not isinstance(body, list) or not all(isinstance(t, str) for t in body):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be a JSON array of strings.")

    return StreamingResponse(event_generator(body), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# Application entry point
# --------------------------------------------------------------------------- #

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI application using Uvicorn."""
    import uvicorn

    logger.info("Starting FastAPI server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # Allow overriding host/port via environment variables for Docker compatibility.
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    start_server(host, port)