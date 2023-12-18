import json
import argparse
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

def _xgboost_regressor(args):
    # Apre e legge il file "data"
    with open(args.data) as data_file:
        data = json.load(data_file)

    # Il tipo di dati atteso è 'dict', tuttavia, poiché il file
    # è stato caricato come un oggetto json, è prima caricato come stringa
    # quindi dobbiamo caricare nuovamente da tale stringa per ottenere
    # l'oggetto di tipo dict.
    data = json.loads(data)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # Inizializza e addestra il modello XGBoost Regressor
    model = xgb.XGBRegressor()

    # Parameter grid for XGBoost Regressor
    model_param_grid = {
        'learning_rate': [0.01, 0.02, 0.03, 0.04],
        'subsample': [0.9, 0.5, 0.2, 0.1],
        'n_estimators': [100, 500, 1000, 1500],
        'max_depth': [4, 6, 8, 10]
    }

    # Perform GridSearch
    grid = GridSearchCV(model, model_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    best_params = str(grid.best_params_)
    best_mae = grid.best_score_

    # Salva l'output nel file
    with open(args.mae, 'w') as mae_file:
        mae_file.write(str(best_mae))

    # Salva l'output nel file
    with open(args.params, 'w') as params_file:
        params_file.write(str(best_params))

if __name__ == '__main__':
    # Definizione e analisi degli argomenti da riga di comando
    parser = argparse.ArgumentParser(description='Descrizione del mio programma')
    parser.add_argument('--data', type=str)
    parser.add_argument('--mae', type=str)
    parser.add_argument('--params', type=str)

    args = parser.parse_args()

    # Creazione della directory in cui verrà creato il file di output (la directory potrebbe o potrebbe non esistere).
    Path(args.mae).parent.mkdir(parents=True, exist_ok=True)
    Path(args.params).parent.mkdir(parents=True, exist_ok=True)

    _xgboost_regressor(args)