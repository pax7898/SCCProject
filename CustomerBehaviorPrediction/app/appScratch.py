import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump

# QUESTO SERVE SOLO PER GENERARE GLI SCALERS

def load_dataset():
    df = pd.read_csv('CustomerBehaviorPrediction/pipeline/load_data/ecommerce_customers.csv')
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

test_size = 0.2
random_state = 50

X = df.drop('Yearly Amount Spent', axis=1)
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

columns = X.columns
scaler = StandardScaler()
scaler = scaler.fit(X)
x = scaler.transform(X)
features = pd.DataFrame(x, columns = columns)
dump(scaler, 'CustomerBehaviorPrediction/app/scalers/data_scaler.joblib')
print("\n DATA SCALER GENERATO")

model = train_model(X_train, y_train)
y_pred = testing_model(model, X_test, y_test)
eval = evaluate(y_pred, y_test)
print(eval)
dump(model, 'CustomerBehaviorPrediction/app/scalers/model_scaler.joblib')
print("\n MODEL SCALER GENERATO")


