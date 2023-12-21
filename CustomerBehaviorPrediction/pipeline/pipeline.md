## Pipeline

### Install kubeflow on kubernetes cluster
```bash
export PIPELINE_VERSION=2.0.3
```

```bash
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
```

```bash
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
```

```bash
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
```

### Enable dashboard on localhost

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Link to dashboard:
http://localhost:8080/

### Upload and execute `customers_pipeline.yaml` in kubeflow.
