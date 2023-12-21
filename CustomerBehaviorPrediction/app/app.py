import streamlit as st
import pandas as pd
import joblib

@st.cache_data()
def load(scaler_path, model_path):
    sc = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return sc, model


def inference(row, scaler, model, feat_cols):
    df = pd.DataFrame([row], columns = feat_cols)
    X = scaler.transform(df)
    features = pd.DataFrame(X, columns = feat_cols)
    prediction = model.predict(features)

    return prediction

st.title('Forecast of a customer\'s annual expenses')
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
sc, model = load('models/data_scaler.joblib',
                 'models/model.joblib')
result = inference(row, sc, model, feat_cols)[0]

if (st.button("Show Result")):
    st.header("This predicted Amount Spent: ${}".format(int(result)))