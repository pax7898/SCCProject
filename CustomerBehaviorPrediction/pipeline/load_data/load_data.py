import json
import argparse
import pandas as pd
from pathlib import Path


# Load Data component:
# - carica i dati da file esterni
# - rimuove feature non rilevanti 
# - prepara la struttura dati per il preprocessing

def _load_data(args):
    # Check for retraining
    with open(args.retrain, 'r') as file:
        retrain = file.read()
    if (retrain):
        trainset=args.train_set
    else:
        trainset='dataset1.csv'

    # Training Data
    with open(trainset, 'r') as file:
        train_set = file.read()
    train_df = pd.read_csv(train_set)
    train_df.drop(['Email', 'Address', 'Avatar'], axis=1, inplace=True)
    train_data = train_df.to_numpy().tolist()
    train_data_json = json.dumps(train_data)
    with open(args.train_raw_data, 'w') as out_file:
        json.dump(train_data_json, out_file)

    # Test Data
    with open(args.test_set, 'r') as file:
        test_set = file.read()
    test_df = pd.read_csv(test_set)
    test_df.drop(['Email', 'Address', 'Avatar'], axis=1, inplace=True)
    test_data = test_df.to_numpy().tolist()
    test_data_json = json.dumps(test_data)
    with open(args.test_raw_data, 'w') as out_file:
        json.dump(test_data_json, out_file)

if __name__ == '__main__':
    # This component does not receive any input
    # it only outputs one artifact which is `data`.
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', type=str)
    parser.add_argument('--test_set', type=str)
    parser.add_argument('--retrain', type=str)
    parser.add_argument('--train_raw_data', type=str)
    parser.add_argument('--test_raw_data', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created 
    # (the directory may or may not exist).
    Path(args.train_raw_data).parent.mkdir(parents=True, exist_ok=True)
    Path(args.test_raw_data).parent.mkdir(parents=True, exist_ok=True)

    _load_data(args)
