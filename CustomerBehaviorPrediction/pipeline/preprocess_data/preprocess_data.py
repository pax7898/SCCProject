import json
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# DA INTEGRARE CON customers_pipeline 

def _preprocess_data(args):
    # Apri e leggi i files JSON creato da load_data.py
    print("Train Raw Data Path:", args.train_raw_data)
    print("Test Raw Data Path:", args.test_raw_data)

    with open(args.train_raw_data) as data_file:
        train_data = json.load(data_file)
    with open(args.test_raw_data) as data_file:
        test_data = json.load(data_file)

    train_data_loaded = json.loads(train_data)
    test_data_loaded = json.loads(test_data)

    # Crea un DataFrame per le features e un Series per il target
    train_df = pd.DataFrame(train_data_loaded, columns=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership','Yearly Amount Spent'])
    test_df = pd.DataFrame(test_data_loaded, columns=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership','Yearly Amount Spent'])

    # Gestione dei valori mancanti con la mediana
    imputer = SimpleImputer(strategy='median')
    train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
    test_df = pd.DataFrame(imputer.fit_transform(test_df), columns=test_df.columns)

    # Standardizzazione delle features numeriche
    scaler = StandardScaler()
    numeric_features = train_df.columns[:-1]  # Escludi l'ultima colonna (la variabile target)
    train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
    test_df[numeric_features] = scaler.fit_transform(test_df[numeric_features])

    # Aumento del dataset per il training(aumento di 200 entries)
    train_df_augmented = train_df.sample(n=70, replace=True, random_state=42)
    train_df = pd.concat([train_df, train_df_augmented], ignore_index=True)

    y_train = train_df['Yearly Amount Spent']
    x_train = train_df.drop('Yearly Amount Spent', axis=1)
    y_test = test_df['Yearly Amount Spent']
    x_test = test_df.drop('Yearly Amount Spent', axis=1)

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
    parser.add_argument('--train_raw_data', type=str)
    parser.add_argument('--test_raw_data', type=str)
    parser.add_argument('--data', type=str)

    args = parser.parse_args()

    Path(args.data).parent.mkdir(parents=True, exist_ok=True)

    _preprocess_data(args)