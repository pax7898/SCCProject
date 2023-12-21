import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
import xgboost as xgb


###########################################################################################
### SCRIPT PER LA CREAZIONE DELLO SCALER E DEL MODELLO UTILIZZATO DALLA WEB APPLICATION ###
###########################################################################################

def load_dataset():
    df = pd.read_csv('../pipeline/load_data/ecommerce_customers.csv')
    return df


def pre_processing(df, test_size, random_state):
    df.drop(['Email', 'Address', 'Avatar'], axis=1, inplace=True)
    y = df['Yearly Amount Spent']
    X = df.drop('Yearly Amount Spent', axis=1)

    columns = X.columns
    scaler = StandardScaler()
    scaler = scaler.fit(X)
    x = scaler.transform(X)
    features = pd.DataFrame(x, columns=columns)
    dump(scaler, 'models/data_scaler.joblib')
    print("\n DATA SCALER DUMP GENERATED")

    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def train_model(xgb, X_train, y_train, params):
    xgb.set_params(learning_rate=params['learning_rate'], max_depth=params['max_depth'],
                   n_estimators=params['n_estimators'], subsample=params['subsample'])
    xgb.fit(X_train, y_train)
    dump(xgb, 'models/model.joblib')
    print("\n MODEL DUMP GENERATED")
    return xgb


def testing_model(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def evaluate(y_pred, y_test):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    e = ['MAE', 'RMSE', 'R-Squared']
    eval = pd.DataFrame([mae, rmse, r2], index=e, columns=['Score'])
    return eval


if __name__ == '__main__':
    df = load_dataset()

    X_train, X_test, y_train, y_test = pre_processing(df, test_size=0.2, random_state=50)

    xgb_best_params = {'learning_rate': 0.04, 'max_depth': 10, 'n_estimators': 1500, 'subsample': 0.2}
    model = train_model(xgb.XGBRegressor(), X_train, y_train, xgb_best_params)

    y_pred = testing_model(model, X_test)
    eval = evaluate(y_pred, y_test)

    print(eval)
