docker build --tag load_data load_data/.
docker tag load_data pax7898/load_data
docker push docker.io/pax7898/load_data

docker build --tag linear_regression linear_regression/.
docker tag linear_regression pax7898/linear_regression
docker push docker.io/pax7898/linear_regression

docker build --tag xgboost_regressor xgboost_regressor/.
docker tag xgboost_regressor pax7898/xgboost_regressor
docker push docker.io/pax7898/xgboost_regressor

# docker build --tag neural_regression neural_regression/.
# docker tag neural_regression pax7898/neural_regression
# docker push docker.io/pax7898/neural_regression