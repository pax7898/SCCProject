import json
import argparse
import pandas as pd
from pathlib import Path


# Load Data component:
# - carica i dati da file esterni
# - rimuove feature non rilevanti 
# - prepara la struttura dati per il preprocessing

# Training Data
with open('data/dataset1.csv', 'r') as file:
    train_set = file.read()
    print(train_set)
train_df = pd.read_csv(train_set)
train_df.drop(['Email', 'Address', 'Avatar'], axis=1, inplace=True)
train_data = train_df.to_numpy().tolist()
train_data_json = json.dumps(train_data)
with open('MYDS', 'w') as out_file:
    json.dump(train_data_json, out_file)

# Test Data
with open('data/testset.csv', 'r') as file:
    test_set = file.read()
    print(test_set)
test_df = pd.read_csv(test_set)
test_df.drop(['Email', 'Address', 'Avatar'], axis=1, inplace=True)
test_data = test_df.to_numpy().tolist()
test_data_json = json.dumps(test_data)
with open('MYTS', 'w') as out_file:
    json.dump(test_data_json, out_file)
