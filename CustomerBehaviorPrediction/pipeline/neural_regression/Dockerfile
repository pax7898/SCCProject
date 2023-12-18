FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements_nr.txt /pipeline
RUN pip install -r requirements_nr.txt
COPY neural_regression.py /pipeline
