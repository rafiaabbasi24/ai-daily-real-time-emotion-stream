# Real-Time Multilingual Emotion Stream Analyzer

## 📖 Project Overview

The **Real-Time Multilingual Emotion Stream Analyzer** is a production‑ready system that ingests live text streams (e.g., social media feeds, chat logs, or customer support tickets), detects the language, and predicts the underlying emotion in real time.  

Key features:

- **Multilingual support** – models for 30+ languages powered by HuggingFace Transformers.
- **Online learning** – incremental model updates with River for concept drift adaptation.
- **Low‑latency inference** – optimized PyTorch inference pipeline wrapped in a FastAPI service.
- **Containerized deployment** – Dockerfile and Docker Compose for reproducible, scalable deployment.
- **Extensible architecture** – plug‑and‑play components for custom preprocessing, model selection, or output sinks.

The system is built with Python 3.10+, PyTorch, HuggingFace Transformers, River, and FastAPI, making it suitable for both research prototyping and production environments.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Modeling** | PyTorch, HuggingFace Transformers | Pre‑trained multilingual language models (e.g., `xlm-roberta-base`) fine‑tuned for emotion classification |
| **Online Learning** | River | Incremental updates on streaming data to handle concept drift |
| **API** | FastAPI | Asynchronous HTTP endpoint for low‑latency predictions |
| **Containerization** | Docker, Docker Compose | Reproducible builds, easy scaling |
| **Utilities** | Python 3.10+, `pydantic`, `uvicorn`, `loguru` | Data validation, server runtime, logging |
| **Testing** | pytest, httpx | End‑to‑end API tests |

---

## 📦 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/real-time-multilingual-emotion-stream-analyzer.git
cd real-time-multilingual-emotion-stream-analyzer
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.\.venv\Scripts\activate    # Windows
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: The `requirements.txt` file pins the exact versions used in production. If you need GPU acceleration, install `torch` with the appropriate CUDA version (e.g., `pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html`).

### 4. Build the Docker image (optional)

```bash
docker compose build
```

### 5. Verify the installation

```bash
pytest -q
```

All tests should pass (≈ 30 tests covering model loading, streaming pipeline, and API endpoints).

---

## 🚀 Usage

### 1. Start the service

#### Local development (Python)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Production (Docker)

```bash
docker compose up -d
```

The API will be available at `http://localhost:8000`.

### 2. API Endpoints

| Method | Path | Description | Example Request |
|--------|------|-------------|-----------------|
| `POST` | `/predict` | Predict emotion for a single text snippet. | `{ "text": "I love this product!" }` |
| `POST` | `/stream` | Submit a batch of texts (max 100) for bulk prediction. | `{ "texts": ["I am sad.", "¡Qué alegría!"] }` |
| `GET` | `/health` | Health check – returns `200 OK` if the service is running. | – |
| `POST` | `/learn` | Send a labeled example to update the online model (`River`). | `{ "text": "I feel terrible.", "label": "sadness" }` |

#### Example with `curl`

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"I am thrilled about the new release!"}'
```

**Response**

```json
{
  "language": "en",
  "emotion": "joy",
  "confidence": 0.93
}
```

### 3. Streaming Integration

The library provides a convenient `StreamProcessor` class that can be used in any Python streaming framework (Kafka, RabbitMQ, etc.). Below is a minimal example using the built‑in generator:

```python
from app.pipeline import StreamProcessor

processor = StreamProcessor()
for raw_text in ["I hate waiting.", "¡Me encanta!"]:
    result = processor.process(raw_text)
    print(result)
```

**Output**

```
{'language': 'en', 'emotion': 'anger', 'confidence': 0.88}
{'language': 'es', 'emotion': 'joy', 'confidence': 0.91}
```

### 4. Updating the Model Online

When new labeled data becomes available, send it to the `/learn` endpoint or call the `online_update` method directly:

```python
from app.online import OnlineUpdater

updater = OnlineUpdater()
updater.online_update(text="I'm feeling anxious.", label="fear")
```

The underlying River model will incorporate the example without retraining from scratch.

---

## 📁 Repository Structure

```
real-time-multilingual-emotion-stream-analyzer/
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI entry point
│   ├── models.py             # PyTorch model wrapper
│   ├── pipeline.py           # End‑to‑end processing (lang detection, inference, post‑proc)
│   ├── online.py             # River online learner utilities
│   └── schemas.py            # Pydantic request/response models
├── tests/
│   ├── test_api.py
│   ├── test_pipeline.py
│   └── test_online.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md                # ← you are here
```

---

## 🧪 Testing

Run the full test suite with:

```bash
pytest -vv
```

Key test groups:

- **Unit tests** – model loading, language detection, and confidence scaling.
- **Integration tests** – FastAPI endpoints using `httpx.AsyncClient`.
- **Performance sanity** – ensures inference latency stays < 100 ms per request on CPU.

---

## 📈 Monitoring & Logging

- **Logging** – `loguru` writes structured JSON logs to `stdout`. Docker logs can be collected with any log aggregation tool (e.g., ELK, Loki).
- **Metrics** – Prometheus exporter is exposed at `/metrics` (requires `prometheus_fastapi_instrumentator` – installed via `requirements.txt`). Track request latency, error rates, and online model update counts.

---

## ⚙️ Configuration

All runtime options are configurable via environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | HuggingFace model identifier (e.g., `xlm-roberta-base`) | `xlm-roberta-base` |
| `MAX_BATCH_SIZE` | Maximum number of texts per `/stream` request | `100` |
| `LOG_LEVEL` | Loguru log level (`DEBUG`, `INFO`, …) | `INFO` |
| `PORT` | FastAPI listening port | `8000` |
| `WORKERS` | Number of Uvicorn workers (for production) | `4` |

Docker Compose automatically loads `.env` at startup.

---

## 📦 Deployment

### Cloud (AWS ECS / Azure Container Apps)

1. Build the image and push to a container registry:

   ```bash
   docker build -t your-registry/rt-emotion-analyzer:latest .
   docker push your-registry/rt-emotion-analyzer:latest
   ```

2. Deploy using your orchestration platform, ensuring the environment variables are set and the container has at least 2 CPU cores and 4 GB RAM for acceptable latency.

### Kubernetes (Helm)

A minimal Helm chart is provided under `helm/`. Install with:

```bash
helm install rt-emotion-analyzer ./helm
```

Configure values via `values.yaml` (replica count, resources, autoscaling).

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/your-feature`).
3. Write tests for new functionality.
4. Ensure `pytest` passes and code is linted (`ruff` or `flake8`).
5. Submit a Pull Request with a clear description of changes.

Please adhere to the **PEP 8** style guide and include docstrings for all public functions/classes.

---

## 📜 License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## 📞 Contact

- **Maintainer**: Jane Doe – `jane.doe@example.com`
- **Project board**: https://github.com/your-org/real-time-multilingual-emotion-stream-analyzer/projects/1

Feel free to open an issue for bugs, feature requests, or general questions.