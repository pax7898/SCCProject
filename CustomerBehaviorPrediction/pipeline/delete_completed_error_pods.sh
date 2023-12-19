#!/bin/bash

kubectl get pods --all-namespaces | awk '$4=="Completed" || $4=="Error" {print $2}' | xargs -I {} kubectl delete pod {} --namespace=kubeflow
