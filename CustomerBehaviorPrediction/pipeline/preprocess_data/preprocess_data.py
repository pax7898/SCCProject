import json
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump


def _preprocess_data(args):
    # Open and read JSON files created by the load_data component
    with open(args.train_raw_data) as data_file:
        train_data = json.load(data_file)
    with open(args.test_raw_data) as data_file:
        test_data = json.load(data_file)

    train_data_loaded = json.loads(train_data)
    test_data_loaded = json.loads(test_data)

    # Create a DataFrame for the features
    train_df = pd.DataFrame(train_data_loaded, columns=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership','Yearly Amount Spent'])
    test_df = pd.DataFrame(test_data_loaded, columns=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership','Yearly Amount Spent'])

    # Handling missing values with the median
    imputer = SimpleImputer(strategy='median')
    train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
    test_df = pd.DataFrame(imputer.fit_transform(test_df), columns=test_df.columns)

    # Standardization of numerical features.
    scaler = StandardScaler()
    numeric_features = train_df.columns[:-1]  # Escludi l'ultima colonna (la variabile target)
    train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
    test_df[numeric_features] = scaler.fit_transform(test_df[numeric_features])

    # Saves the dump of the scaler into a file
    dump(scaler, args.scaler)

    # Data augmentation for training
    train_df_augmented = train_df.sample(n=70, replace=True, random_state=42)
    train_df = pd.concat([train_df, train_df_augmented], ignore_index=True)

    y_train = train_df['Yearly Amount Spent']
    x_train = train_df.drop('Yearly Amount Spent', axis=1)
    y_test = test_df['Yearly Amount Spent']
    x_test = test_df.drop('Yearly Amount Spent', axis=1)

    x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    # Creates `data` structure to save and share train and test datasets.
    data = {'x_train': x_train.tolist(), 'y_train': y_train.tolist(), 'x_test': x_test.tolist(), 'y_test': y_test.tolist()}

    # Creates a json object based on `data`
    data_json = json.dumps(data)

    # Saves the json object into a file
    with open(args.data, 'w') as out_file:
        json.dump(data_json, out_file)


if __name__ == '__main__':
    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_raw_data', type=str)
    parser.add_argument('--test_raw_data', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--scaler', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created
    # (the directory may or may not exist).
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scaler).parent.mkdir(parents=True, exist_ok=True)

    _preprocess_data(args)