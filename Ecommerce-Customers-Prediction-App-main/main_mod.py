import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures


st.title('Predicting Customer Spent')
st.markdown("""
            This is an application to find out how much customers spend on e-commerce in one year!
        
            Dataset from [Kaggle](https://www.kaggle.com/srolka/ecommerce-customers)
        """)

def load_dataset():
    df = pd.read_csv('Ecommerce Customers.csv')
    return df

def train_model(X_train, y_train):
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    return linear_regression

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
df.drop(['Email', 'Address', 'Avatar'], axis=1, inplace=True)

# Crea la struttura dati per salvare e condividere i dataset
data = df.to_numpy().tolist()

# Crea un oggetto JSON basato su `data`
data_json = json.dumps(data)

# Salva l'oggetto JSON in un file
with open("TEST_DUMP.json", 'w') as out_file:
    out_file.write(data_json)

# Apri e leggi il file JSON creato da load_data.py
with open("TEST_DUMP.json") as data_file:
    data_loaded = json.load(data_file)

# # Il tipo di dati atteso è list
# print(data)

# Crea un DataFrame per le features e un Series per il target
df = pd.DataFrame(data_loaded, columns=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership','Yearly Amount Spent'])

# Gestione dei valori mancanti con la mediana
imputer = SimpleImputer(strategy='median')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Standardizzazione delle features numeriche
scaler = StandardScaler()
numeric_features = df.columns[:-1]  # Escludi l'ultima colonna (la variabile target)
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Aumento del dataset (aumento di 200 entries)
df_augmented = df.sample(n=200, replace=True, random_state=42)
df = pd.concat([df, df_augmented], ignore_index=True)

x = df.drop('Yearly Amount Spent', axis=1)
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


st.header("Dataset")
st.write(df)
# df.drop(['Email', 'Address', 'Avatar'], axis=1, inplace=True)

# st.header("Tune Parameters")
# st.write("Let's find out how much the result when we choosing some parameters")
# test_size = st.slider('Test Size', 0.1, 0.5, 0.1)
# random_state = st.slider('Random State', 0, 200, 1)

# X = df.drop('Yearly Amount Spent', axis=1)
# y = df['Yearly Amount Spent']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

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


# Standardizzazione delle features numeriche
new_input_scaled = scaler.transform([[avg_session_length, time_app, time_web, length_member]])

# Predizione con il modello
predictions = model.predict(new_input_scaled)

# Inversa della standardizzazione per ottenere il risultato originale
predicted_amount_spent = predictions * scaler.scale_[-1] + scaler.mean_[-1]

if (st.button("Show Result")):
    st.header("This predicted Amount Spent: ${}".format(int(predicted_amount_spent)))
