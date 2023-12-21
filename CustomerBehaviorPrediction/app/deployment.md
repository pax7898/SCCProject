<h2>Customer behavior prediction app on Kubernetes cluster </h2>

<h3>Deployment</h3>

Run:
```bash
kubectl create --filename k8s_customer_behavior_deployment.yaml
```

<h3> Use </h3>

Visit the following URL: http://127.0.0.1:30070/

<h3> Clean </h3>
To delete the deployment run:

```bash
kubectl delete services customer-behavior && kubectl delete deployment customer-behavior
```

