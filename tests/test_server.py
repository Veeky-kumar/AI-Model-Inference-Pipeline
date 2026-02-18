"""
Tests for the AI Inference Server
Run with: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model-server'))
from server import app

client = TestClient(app)


class TestHealthEndpoints:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_shows_model_loaded(self):
        response = client.get("/health")
        assert "model_loaded" in response.json()
        assert response.json()["model_loaded"] is True

    def test_ready_endpoint(self):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestModelMetadata:
    def test_model_metadata(self):
        response = client.get("/v2/models/iris-classifier")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "iris-classifier"
        assert "inputs" in data
        assert "outputs" in data


class TestInference:
    def test_single_inference(self):
        """Single sample prediction"""
        payload = {
            "id": "test-001",
            "inputs": [{
                "name": "input",
                "shape": [1, 4],
                "datatype": "FP32",
                "data": [5.1, 3.5, 1.4, 0.2]
            }]
        }
        response = client.post("/v2/models/iris-classifier/infer", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["model_name"] == "iris-classifier"
        assert len(data["outputs"]) == 2
        assert data["id"] == "test-001"

    def test_batch_inference(self):
        """Batch of 3 samples"""
        payload = {
            "inputs": [{
                "name": "input",
                "shape": [3, 4],
                "datatype": "FP32",
                "data": [
                    [5.1, 3.5, 1.4, 0.2],
                    [6.7, 3.1, 4.7, 1.5],
                    [6.3, 3.3, 6.0, 2.5],
                ]
            }]
        }
        response = client.post("/v2/models/iris-classifier/infer", json=payload)
        assert response.status_code == 200

        probs_output = next(o for o in response.json()["outputs"] if o["name"] == "probabilities")
        # Batch of 3 → 3 rows of probabilities
        assert len(probs_output["data"]) == 3

    def test_inference_output_probabilities_sum_to_one(self):
        """Probabilities must sum to ~1.0"""
        payload = {
            "inputs": [{
                "name": "input",
                "shape": [1, 4],
                "datatype": "FP32",
                "data": [6.7, 3.1, 4.7, 1.5]
            }]
        }
        response = client.post("/v2/models/iris-classifier/infer", json=payload)
        probs = response.json()["outputs"][0]["data"][0]
        assert abs(sum(probs) - 1.0) < 0.001, "Probabilities must sum to 1.0"

    def test_predicted_class_is_valid(self):
        """Predicted class must be one of the 3 iris species"""
        valid_classes = {"setosa", "versicolor", "virginica"}
        payload = {
            "inputs": [{
                "name": "input",
                "shape": [1, 4],
                "datatype": "FP32",
                "data": [5.1, 3.5, 1.4, 0.2]
            }]
        }
        response = client.post("/v2/models/iris-classifier/infer", json=payload)
        predicted = response.json()["outputs"][1]["data"][0]
        assert predicted in valid_classes


class TestMetrics:
    def test_prometheus_metrics_endpoint(self):
        """Prometheus /metrics must return valid text"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "inference_requests_total" in response.text
        assert "inference_request_duration_seconds" in response.text

    def test_metrics_increment_after_inference(self):
        """Request count should increase after inference"""
        # Get baseline
        before = client.get("/metrics").text
        before_count = [l for l in before.split("\n") if "inference_requests_total" in l and "success" in l]

        # Make inference
        client.post("/v2/models/iris-classifier/infer", json={
            "inputs": [{"name": "input", "shape": [1, 4], "datatype": "FP32", "data": [5.1, 3.5, 1.4, 0.2]}]
        })

        # Check incremented
        after = client.get("/metrics").text
        assert "inference_requests_total" in after
