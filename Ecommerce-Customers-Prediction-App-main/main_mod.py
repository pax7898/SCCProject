import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title('Predicting Customer Spent')
st.markdown("""
            This is an application to find out how much customers spend on e-commerce in one year!
        
            Dataset from [Kaggle](https://www.kaggle.com/srolka/ecommerce-customers)
        """)

def load_dataset():
    df = pd.read_csv('Ecommerce Customers.csv')
    return df

def train_model(X_train, y_train):
    neural_network_regressor = MLPRegressor(random_state=1, max_iter=500)
    neural_network_regressor.fit(X_train, y_train)
    return neural_network_regressor

def testing_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return y_pred

def evaluate(y_pred, y_test):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    e = ['MAE', 'RMSE', 'R-Squared']
    eval = pd.DataFrame([mae, rmse, r2], index=e, columns=['Score'])
    return eval

df = load_dataset()

st.header("Dataset")
st.write(df)
df.drop(['Email', 'Address', 'Avatar'], axis=1, inplace=True)

st.header("Tune Parameters")
st.write("Let's find out how much the result when we choosing some parameters")
test_size = st.slider('Test Size', 0.1, 0.5, 0.1)
random_state = st.slider('Random State', 0, 200, 1)

X = df.drop('Yearly Amount Spent', axis=1)
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

st.header('Model Performance')
model = train_model(X_train, y_train)
y_pred = testing_model(model, X_test, y_test)
eval = evaluate(y_pred, y_test)
st.write(eval)

st.header('Predict New Value')
avg_session_length = st.slider('Average Session Length', 0, 90, 1)
time_app = st.slider('Time on App', 0, 90, 1)
time_web = st.slider('Time on Web', 0, 90, 1)
length_member = st.slider('Length of Membership', 0, 10, 1)

predictions = model.predict([[avg_session_length, time_app, time_web, length_member]])

if (st.button("Show Result")):
    st.header("This predicted Amount Spent: ${}".format(int(predictions)))
