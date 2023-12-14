FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements_ld.txt /pipeline
RUN pip install -r requirements_ld.txt
COPY load_data.py /pipeline
COPY diabetes.csv /pipeline