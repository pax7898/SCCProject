<h2> Project Work: Customer Behavior Prediction</h2>

Il progetto prevede la realizzazione di una wep app per la previsione delle spese annue future dei clienti di un servizio di e-commerce.

Il training e il testing del modello di ML è stato effettuato mediante una pipeline Kubeflow.

In particolare il codice all'interno della directory corrente CustomerBehaviorPrediction è diviso in quattro sotto directory:

- `/app`
- `/cluster`
- `/pipeline`
- `/NGINX_example`

<h3> App </h3>

All'interno della directory `/app` sono presenti:

- `app.py`: codice dell'applicazione web 
- `Dockerfile`: dockerfile per la creazione del container dell'applicazione
- `k8s_customer_behavior_deployment.yaml`: file per la configurazione dei pods relativi all'applicazione nel cluster (Deployment & Service)
- `requirements.txt`: librerie python necessarie all'esecuzione dell'applicazione che verranno inserite nel container
- `/model`: directory in cui sono contenuti lo scaler e il modello utilizzati dall'applicazione
- `deployment.md`: file che illustra come effettuare il deployment e il run dell'applicazione sul cluster
- `docker_hub.sh`: script per la creazione del container dell'applicazione e per il push dell'immagine su docker hub

<h3> Cluster </h3>

All'interno della directory `/cluster` sono presenti:

- `cluster_config.yaml`: file di configurazione del cluster Kubernetes
- `metric-server.yaml`: file di configurazione del metric-server
- `cluster.md`: file che illustra come effettuare il setup del cluster e come installare il metric-server

<h3> Pipeline </h3>

All'interno della directory `/pipeline` sono presenti:

- `/load_data`: Component che si occupa del load del dataset per il training
- `/preprocess_data`: Component che si occupa del preprocessing dei dati
- `/linear_regression`: Component che si occupa del training e testing del modello Linear Regression
- `/xgboost_regressor`: Component che si occupa del training e testing del modello XGBoost Regressor
- `/neural_regression`: Component che si occupa del training e testing del modello MLP
- `customer_pipeline.py`: file per la generazione del file .yaml della pipeline kubeflow (contiene anche la definizione dei component show_result e evaluate_best_model)
- `customer_pipeline.yaml`: file .yaml della pipeline da inserire in kubeflow
- `docker_hub.sh`: script per la creazione dei container dei vari component e per il push delle immagini su docker hub
- `pipeline.md`: file che illustra come installare kubeflow sul kubernetes cluster ed importare la pipeline

<h3> NGINX_example </h3>

All'interno della directory `/NGINX_example` sono presenti:

- `IngressController.yaml`: file di configurazione dell'ingress controller per il Web App Firewall
- `VirtualServer.yaml`: file di configurazione del virtual server del Web App Firewall