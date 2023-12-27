## To execute pipeline
After Kubernetes cluster allocation follow these steps:
1. ### Install kubeflow on kubernetes cluster
    Install Kubeflow: The PIPELINE_VERSION environment variable is set, and Kubeflow Pipelines are installed on the Kubernetes cluster using kubectl apply -k. This step deploys all necessary resources for Kubeflow.

    ```bash
    export PIPELINE_VERSION=2.0.3
    ```

    ```bash
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
    ```

    Wait for CRD Establishment: The command kubectl wait ensures that the Custom Resource Definitions (CRDs) are established before proceeding.

    ```bash
    kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
    ```

    ```bash
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
    ```

3. ### Enable dashboard on localhost

    Dashboard Setup: The kubectl port-forward command forwards a local port to the Kubeflow dashboard service running in the cluster, allowing access to the Kubeflow dashboard via http://localhost:8080/.

    ```bash
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
    ```

4. ### Upload and execute `customers_pipeline.yaml` in kubeflow.