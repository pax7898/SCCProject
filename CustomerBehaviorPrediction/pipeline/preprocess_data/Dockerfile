FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements_pd.txt /pipeline
RUN pip install -r requirements_pd.txt
COPY preprocess_data.py /pipeline