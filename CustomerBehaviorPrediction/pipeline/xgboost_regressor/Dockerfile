FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements_xgbr.txt /pipeline
RUN pip install -r requirements_xgbr.txt
COPY xgboost_regressor.py /pipeline
COPY xgboost_regressor.joblib /pipeline