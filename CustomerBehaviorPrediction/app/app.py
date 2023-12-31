import streamlit as st
import pandas as pd
import joblib


@st.cache_data()
def load(scaler_path, model_path):
    sc = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return sc, model


def inference(row, scaler, model, feat_cols):
    df = pd.DataFrame([row], columns=feat_cols)
    X = scaler.transform(df)
    features = pd.DataFrame(X, columns=feat_cols)
    prediction = model.predict(features)

    return prediction


st.title('Predict customer\'s annual expenses')
st.markdown("""
            Application to find out the customers annual expenses on an e-commerce in one year!
        """)

st.header('Predict New Value')
avg_session_length = st.slider('Average Session Length (Minutes)', 0, 90, 1)
time_app = st.slider('Time on App (Minutes)', 0, 90, 1)
time_web = st.slider('Time on Web (Minutes)', 0, 90, 1)
length_member = st.slider('Length of Membership (Years)', 0, 10, 1)

row = [avg_session_length, time_app, time_web, length_member]
feat_cols = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']

sc, model = load('models/scaler.joblib',
                 'models/model.joblib')

result = inference(row, sc, model, feat_cols)[0]

if st.button("Show Result"):
    st.header("Predicted Amount Spent: ${}".format(int(result)))
