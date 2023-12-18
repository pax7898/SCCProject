import json
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# DA INTEGRARE CON load_data e customers_pipeline 
# PER ORA EFFETTUA ANCHE QUI IL LOAD DAL CSV

def _preprocess_data(args):
    # Gets and split dataset
    df = pd.read_csv('ecommerce_customers.csv')
    df.drop(['Email', 'Address', 'Avatar'], axis=1, inplace=True)

    # Gestione dei valori mancanti con la mediana
    imputer = SimpleImputer(strategy='median')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Creazione di nuove features polinomiali
    poly_features = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_data = poly_transformer.fit_transform(df[poly_features])
    poly_columns = poly_transformer.get_feature_names(poly_features)
    df_poly = pd.DataFrame(poly_data, columns=poly_columns)
    df = pd.concat([df, df_poly], axis=1)

    # Standardizzazione delle features numeriche
    scaler = StandardScaler()
    numeric_features = df.columns
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Aumento del dataset (aumento di 200 entries)
    df_augmented = df.sample(n=200, replace=True, random_state=42)
    df = pd.concat([df, df_augmented], ignore_index=True)

    x = df.drop('Yearly Amount Spent', axis=1)
    y = df['Yearly Amount Spent']
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
    parser.add_argument('--data', type=str)

    args = parser.parse_args()

    Path(args.data).parent.mkdir(parents=True, exist_ok=True)

    _preprocess_data(args)
