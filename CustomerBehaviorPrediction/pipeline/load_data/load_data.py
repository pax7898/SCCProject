import json
import argparse
import pandas as pd
from pathlib import Path

# Load Data component: 
# - carica i dati da file esterni
# - rimuove feature non rilevanti 
# - prepara la struttura dati per il preprocessing

def _load_data(args):

    # Gets and split dataset
    df = pd.read_csv('ecommerce_customers.csv')
    df.drop(['Email', 'Address', 'Avatar'], axis=1, inplace=True)

    # Crea la struttura dati per salvare e condividere i dataset
    data = df.to_numpy().tolist()

    # Creates a json object based on `data`
    data_json = json.dumps(data)

    # Saves the json object into a file
    with open(args.raw_data, 'w') as out_file:
        json.dump(data_json, out_file)

if __name__ == '__main__':
    
    # This component does not receive any input
    # it only outputs one artifact which is `data`.
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', type=str)
    
    args = parser.parse_args()
    
    # Creating the directory where the output file will be created 
    # (the directory may or may not exist).
    Path(args.raw_data).parent.mkdir(parents=True, exist_ok=True)

    _load_data(args)
    