<h2> Project Work: Customer Behavior Prediction</h2>

The project involves the development of a web app for predicting the future annual expenses of customers using an e-commerce service.

The training and testing of the ML model were carried out using a Kubeflow pipeline.

In particular, the code within the current directory "CustomerBehaviorPrediction" is divided into four subdirectories:

- `/app`
- `/cluster`
- `/pipeline`
- `/NGINX_example`

<h3> App </h3>

Within the `/app` directory, the following items are present:

- `app.py`: code for the web application
- `Dockerfile`: Dockerfile for creating the application container
- `k8s_customer_behavior_deployment.yaml`: file for configuring the pods related to the application in the cluster (Deployment & Service)
- `requirements.txt`: Python libraries required for running the application, which will be included in the container
- `/model`: directory containing the scaler and model used by the application
- `deployment.md`: file explaining how to deploy and run the application on the cluster
- `docker_hub.sh`: script for creating the application container and pushing the image to Docker Hub

<h3> Cluster </h3>

Within the `/cluster` directory, the following items are present:

- `cluster_config.yaml`: Kubernetes cluster configuration file
- `metric-server.yaml`: metric-server configuration file
- `cluster.md`: file explaining how to set up the cluster and install the metric-server

<h3> Pipeline </h3>

Within the `/pipeline` directory, the following items are present:

- `/load_data`: Component responsible for loading the dataset for training
- `/preprocess_data`: Component responsible for data preprocessing
- `/linear_regression`: Component responsible for training and testing the Linear Regression model
- `/xgboost_regressor`: Component responsible for training and testing the XGBoost Regressor model
- `/neural_regression`: Component responsible for training and testing the MLP model
- `customer_pipeline.py`: file for generating the kubeflow pipeline YAML file (also contains the definition of the show_result and evaluate_best_model components)
- `customer_pipeline.yaml`: kubeflow pipeline YAML file to be inserted into kubeflow
- `docker_hub.sh`: script for creating containers for various components and pushing images to Docker Hub
- `pipeline.md`: file explaining how to install Kubeflow on the Kubernetes cluster and import the pipeline

<h3> NGINX_example </h3>

Within the `/NGINX_example` directory, the following items are present:

- `IngressController.yaml`: configuration file for the Ingress Controller for the Web App Firewall
- `VirtualServer.yaml`: configuration file for the virtual server of the Web App Firewall
