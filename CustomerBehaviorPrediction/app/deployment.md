<h2>Customer behavior prediction app on Kubernetes cluster </h2>

<h3>Deployment</h3>
Deployment and Service Creation:
```bash
kubectl create --filename k8s_customer_behavior_deployment.yaml
```

Horizontal Pod Autoscaler Creation:
```bash
kubectl autoscale deployment customer-behavior --cpu-percent=50 --min=5 --max=15
```

<h3> Use </h3>

Visit the following URL: http://127.0.0.1:30070/

<h3> Clean </h3>
To delete the deployment and related resources run:

```bash
kubectl delete services customer-behavior && kubectl delete deployment customer-behavior && kubectl delete hpa customer-behavior 
```

