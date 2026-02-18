# ⚡ AI Model Inference Pipeline

> A production-grade, Kubernetes-native model serving system with horizontal autoscaling, real-time observability, and zero-downtime deployments.

[![CI/CD](https://github.com/YOUR_USERNAME/ai-inference-pipeline/actions/workflows/deploy.yml/badge.svg)](https://github.com/YOUR_USERNAME/ai-inference-pipeline/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.28-326ce5?logo=kubernetes)](https://kubernetes.io)
[![KServe](https://img.shields.io/badge/KServe-0.11-orange)](https://kserve.github.io/website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🏗️ Architecture

```
                          ┌──────────────────────────────────────────┐
                          │           Kubernetes Cluster             │
                          │                                          │
Client ──► Ingress ──►  KServe InferenceService                     │
                          │       ┌─────────────────────┐           │
                          │       │   Predictor Pods    │           │
                          │       │  ┌───┐ ┌───┐ ┌───┐ │           │
                          │       │  │P1 │ │P2 │ │P3 │ │  ← HPA   │
                          │       │  └───┘ └───┘ └───┘ │  scales  │
                          │       └──────────┬──────────┘  1–10   │
                          │                  │                       │
                          │       ┌──────────▼──────────┐           │
                          │       │ Prometheus + Grafana │           │
                          │       └─────────────────────┘           │
                          └──────────────────────────────────────────┘
```

## ✨ Features

| Feature | Implementation |
|---------|---------------|
| **Model Serving** | FastAPI + KServe V2 inference protocol |
| **Containerization** | Multi-stage Docker build, non-root user |
| **Orchestration** | Kubernetes Deployments + Services |
| **Autoscaling** | HPA (CPU/Memory/custom metrics) + KEDA |
| **Traffic Splitting** | KServe canary deployments |
| **Observability** | Prometheus metrics + Grafana dashboards |
| **CI/CD** | GitHub Actions: test → build → deploy |
| **Zero-downtime** | Rolling updates, readiness/liveness probes |

---

## 🚀 Quick Start

### Option A: Local with Docker Compose (easiest)

```bash
git clone https://github.com/YOUR_USERNAME/ai-inference-pipeline
cd ai-inference-pipeline

# Start model server + Prometheus + Grafana
docker-compose up -d

# Test inference
curl -X POST http://localhost:8080/v2/models/iris-classifier/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input",
      "shape": [1, 4],
      "datatype": "FP32",
      "data": [5.1, 3.5, 1.4, 0.2]
    }]
  }'

# Open Grafana at http://localhost:3000 (admin/admin123)
```

### Option B: Kubernetes (Minikube)

```bash
# 1. Start Minikube
minikube start --memory=8192 --cpus=4

# 2. Enable metrics server (required for HPA)
minikube addons enable metrics-server

# 3. Build & load image
docker build -t ai-inference-pipeline:latest ./model-server
minikube image load ai-inference-pipeline:latest

# 4. Deploy everything
kubectl apply -f k8s/

# 5. Watch pods start up
kubectl get pods -n ai-inference -w

# 6. Port-forward and test
kubectl port-forward svc/model-server-svc 8080:80 -n ai-inference &
curl http://localhost:8080/health
```

### Option C: Full Production Setup (EKS/GKE/AKS)

```bash
# Bootstraps Istio, KServe, Prometheus, KEDA, and your model
chmod +x scripts/setup_cluster.sh
./scripts/setup_cluster.sh
```

---

## 📡 API Reference

### Health Check
```
GET /health  →  {"status": "ok", "model_loaded": true}
GET /ready   →  {"status": "ready"}
```

### Model Metadata
```
GET /v2/models/iris-classifier
```

### Inference
```
POST /v2/models/iris-classifier/infer
Content-Type: application/json

{
  "id": "req-001",
  "inputs": [{
    "name": "input",
    "shape": [1, 4],
    "datatype": "FP32",
    "data": [5.1, 3.5, 1.4, 0.2]
  }]
}
```
**Response:**
```json
{
  "id": "req-001",
  "model_name": "iris-classifier",
  "model_version": "v1.0.0",
  "outputs": [
    {"name": "probabilities", "data": [[0.92, 0.05, 0.03]]},
    {"name": "predicted_class", "data": ["setosa"]}
  ]
}
```

### Prometheus Metrics
```
GET /metrics
```

---

## ⚡ Triggering HPA Autoscaling

```bash
# Run load test (scales pods 1 → 10)
python scripts/load_test.py --url http://localhost:8080 --rps 200 --duration 60

# Watch HPA respond in real time
kubectl get hpa -n ai-inference -w

# Expected output:
# NAME               REFERENCE             TARGETS          MINPODS   MAXPODS   REPLICAS
# model-server-hpa   Deployment/model-srv  78%/70%          1         10        4
```

---

## 🔧 Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## 📁 Project Structure

```
ai-inference-pipeline/
├── model-server/
│   ├── server.py           # FastAPI inference server (KServe V2 protocol)
│   ├── Dockerfile          # Multi-stage, non-root container
│   └── requirements.txt
├── k8s/
│   ├── 00-namespace.yaml   # Namespace with Istio injection
│   ├── 01-deployment.yaml  # Deployment w/ probes & resource limits
│   ├── 02-service.yaml     # ClusterIP + LoadBalancer
│   └── 03-hpa.yaml         # HPA: CPU + memory + custom metrics
├── kserve/
│   └── inferenceservice.yaml  # KServe + KEDA ScaledObject
├── monitoring/
│   └── prometheus.yml      # Prometheus scrape config
├── .github/workflows/
│   └── deploy.yml          # CI: test → build → push → deploy
├── scripts/
│   ├── setup_cluster.sh    # One-shot cluster bootstrap
│   └── load_test.py        # Async load tester to trigger HPA
├── tests/
│   └── test_server.py      # Unit + integration tests
└── docker-compose.yml      # Local dev: server + prometheus + grafana
```

---

## 🔄 Replacing with Your Own Model

1. Edit `model-server/server.py`, find the `FakeClassificationModel` class
2. Replace `self.weights = np.random.randn(...)` with `self.model = torch.load("model/weights.pt")`
3. Replace the `predict()` method body with your model's forward pass
4. Update `CLASSES`, `MODEL_NAME`, input/output shapes

---

## 📊 Key Prometheus Queries for Grafana

```promql
# p95 inference latency
histogram_quantile(0.95, rate(inference_request_duration_seconds_bucket[5m]))

# Requests per second
rate(inference_requests_total{status="success"}[1m])

# Error rate %
100 * rate(inference_requests_total{status="error"}[1m])
    / rate(inference_requests_total[1m])

# Active pod count
count(kube_pod_info{namespace="ai-inference"})
```

---

## 🏆 Resume Bullet Points (copy-paste ready)

> **AI Model Inference Pipeline** — Built a Kubernetes-based deployment for AI model serving using Docker and KServe. Implemented horizontal pod autoscaling (HPA) to handle fluctuating inference traffic, scaling from 1 to 10 replicas based on CPU utilization and custom Prometheus metrics. Designed a FastAPI server following the KServe V2 inference protocol with full observability (Prometheus/Grafana) and zero-downtime rolling deployments via GitHub Actions CI/CD.

---

## 📚 Learning Resources

| Topic | Resource |
|-------|----------|
| Docker | [docs.docker.com](https://docs.docker.com/get-started/) |
| Kubernetes | [kubernetes.io/tutorials](https://kubernetes.io/docs/tutorials/) |
| KServe | [kserve.github.io](https://kserve.github.io/website/0.11/get_started/) |
| HPA | [K8s HPA Walkthrough](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/) |
| KEDA | [keda.sh/docs](https://keda.sh/docs/2.12/getting-started/) |
| Prometheus | [prometheus.io/docs](https://prometheus.io/docs/prometheus/latest/getting_started/) |
| FastAPI | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/tutorial/) |

---

## License

MIT — use freely for learning and production.
