<h2> Kubernetes Cluster Configuration </h2>

<h3>Cluster Configuration</h3>

Open terminal window and move inside the directory **/CustomerBehaviorPrediction** e run:
```bash
kind create cluster --config=cluster/cluster_config.yaml
```

<h2> Enable and access to Kubernetes Dashboard </h2>

<h3>Installing Kubernetes Dashboard to get the Dashboard</h3>
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml 
```

<h3>Create a service account for the dashboard</h3>
```bash
kubectl apply -f dashboard-adminuser.yaml
```

```bash
kubectl apply -f cluster_rolebinding.yaml
```

<h3>Access to the dashboard with a token</h3>
```bash
kubectl -n kubernetes-dashboard create token admin-user
```
```bash
kubectl proxy
```

Link to dashboard:
http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/