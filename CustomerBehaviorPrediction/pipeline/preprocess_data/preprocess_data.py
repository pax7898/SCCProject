import json
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# DA INTEGRARE CON customers_pipeline 

def _preprocess_data(args):
    # Apri e leggi il file JSON creato da load_data.py
    with open(args.raw_data) as data_file:
        data_loaded = json.load(data_file)

    data = json.loads(data_loaded)
    # Crea un DataFrame per le features e un Series per il target
    df = pd.DataFrame(data, columns=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership','Yearly Amount Spent'])

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

    y = df['Yearly Amount Spent']
    x = df.drop('Yearly Amount Spent', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    # Creates `data` structure to save and 
    # share train and test datasets.
    data = {'x_train': x_train.tolist(), 'y_train': y_train.tolist(), 'x_test': x_test.tolist(), 'y_test': y_test.tolist()}

    # Creates a json object based on `data`
    data_json = json.dumps(data)

    # Saves the json object into a file
    with open(args.data, 'w') as out_file:
        json.dump(data_json, out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', type=str)
    parser.add_argument('--data', type=str)

    args = parser.parse_args()

    Path(args.data).parent.mkdir(parents=True, exist_ok=True)

    _preprocess_data(args)