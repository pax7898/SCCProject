import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


# SICURAMENTE QUALCHE LIBRERIA ANDRÀ TOLTA

# @st.cache(allow_output_mutation=True) -> deprecated
@st.cache_data()
def load(scaler_path, model_path):
    sc = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return sc, model


def inference(row, scaler, model, feat_cols):
    # df = pd.DataFrame([row], columns = feat_cols)
    # X = scaler.transform(df)
    # features = pd.DataFrame(X, columns = feat_cols)
    predictions = model.predict([row])
    return predictions


# DA MIGLIORARE, NON HO BEN CAPITO NELL'app.py DEL PROF PERCHÉ IN inference FA TUTTO QUEL CASINO
# QUI NON CE N'È BISOGNO PERCIÒ È COMMENTATO


st.title('Predicting Customer Spent')
st.markdown("""
            This is an application to find out how much customers spend on e-commerce in one year!
        
            Dataset from [Kaggle](https://www.kaggle.com/srolka/ecommerce-customers)
        """)

st.header('Predict New Value')
avg_session_length = st.slider('Average Session Length', 0, 90, 1)
time_app = st.slider('Time on App', 0, 90, 1)
time_web = st.slider('Time on Web', 0, 90, 1)
length_member = st.slider('Length of Membership', 0, 10, 1)

row = [avg_session_length, time_app, time_web, length_member]
feat_cols = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
sc, model = load('CustomerBehaviorPrediction/app/scalers/data_scaler.joblib',
                 'CustomerBehaviorPrediction/app/scalers/model_scaler.joblib')
result = inference(row, sc, model, feat_cols)

if (st.button("Show Result")):
    st.header("This predicted Amount Spent: ${}".format(int(result)))
