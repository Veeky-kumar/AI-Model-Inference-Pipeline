#!/usr/bin/env bash
# ── AI Inference Pipeline — Full Kubernetes Setup ─────────────────────────────
# Run this once to bootstrap everything on a fresh cluster.
# Prerequisites: kubectl, helm, a running K8s cluster (EKS/GKE/AKS/Minikube)

set -euo pipefail
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ── 1. Namespace ──────────────────────────────────────────────────────────────
info "Creating namespace..."
kubectl apply -f k8s/00-namespace.yaml

# ── 2. Install Istio (required by KServe) ────────────────────────────────────
info "Installing Istio..."
if ! command -v istioctl &>/dev/null; then
  curl -L https://istio.io/downloadIstio | sh -
  export PATH="$PWD/istio-*/bin:$PATH"
fi
istioctl install --set profile=default -y

# ── 3. Install KServe ─────────────────────────────────────────────────────────
info "Installing KServe..."
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve-cluster-resources.yaml

# ── 4. Install Prometheus Stack ───────────────────────────────────────────────
info "Installing Prometheus + Grafana..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=admin123 \
  --wait

# ── 5. Install KEDA ───────────────────────────────────────────────────────────
info "Installing KEDA..."
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm upgrade --install keda kedacore/keda \
  --namespace keda \
  --create-namespace \
  --wait

# ── 6. Deploy our model ───────────────────────────────────────────────────────
info "Deploying model server..."
kubectl apply -f k8s/

# ── 7. Deploy KServe InferenceService ────────────────────────────────────────
info "Deploying KServe InferenceService..."
kubectl apply -f kserve/inferenceservice.yaml

# ── 8. Wait for rollout ───────────────────────────────────────────────────────
info "Waiting for deployment to be ready..."
kubectl rollout status deployment/model-server -n ai-inference --timeout=120s

# ── 9. Print status ───────────────────────────────────────────────────────────
echo ""
info "✅ Setup complete! Status:"
kubectl get pods -n ai-inference
echo ""
kubectl get hpa -n ai-inference
echo ""
info "📊 Access Grafana:"
echo "   kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring"
echo "   Open: http://localhost:3000 (admin/admin123)"
echo ""
info "🧪 Test the API:"
echo "   kubectl port-forward svc/model-server-svc 8080:80 -n ai-inference"
echo "   curl -X POST http://localhost:8080/v2/models/iris-classifier/infer \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"inputs\": [{\"name\": \"input\", \"shape\": [1,4], \"datatype\": \"FP32\", \"data\": [5.1,3.5,1.4,0.2]}]}'"
echo ""
info "⚡ Trigger HPA scaling:"
echo "   python scripts/load_test.py --url http://localhost:8080 --rps 200 --duration 60"
echo "   kubectl get hpa -n ai-inference -w"
