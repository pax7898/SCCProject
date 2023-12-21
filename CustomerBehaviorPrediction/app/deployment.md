<h2>Customer behavior prediction app on Kubernetes cluster </h2>

<h3>Deployment</h3>

Open terminal window and move inside the directory **/CustomerBehaviorPrediction** e run:
```bash
kubectl create --filename app/k8s_customer_behavior_deployment.yaml
```

<h3> Use </h3>

Visit the following URL: http://127.0.0.1:30070/

<h3> Clean </h3>
To delete the deployment run:

```bash
kubectl delete services customer-behavior
```

```bash
kubectl delete deployment customer-behavior
```
