import json
import argparse
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import joblib

def _xgboost_regressor(args):
    # Apre e legge il file "data"
    with open(args.data) as data_file:
        data = json.load(data_file)

    with open(args.retrain, 'r') as file:
        retrain = file.read()

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

    if not retrain:
        model = xgb.XGBRegressor()
    else:
        model = joblib.load("xgboost_regressor.joblib")

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

    y_pred_train = best_model.predict(x_train)
    y_pred_test = best_model.predict(x_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Save MAE on training set of the best model in a local file
    with open(args.mae_train, 'w') as mae_train_file:
        mae_train_file.write(str(mae_train))

    # Save MAE on test set of the best model in a local file
    with open(args.mae_test, 'w') as mae_test_file:
        mae_test_file.write(str(mae_test))

    # Save the parameters of the best model in a local file
    with open(args.params, 'w') as params_file:
        params_file.write(str(best_params))

    # Save the dump of the best model in a local file
    joblib.dump(best_model, args.model)

if __name__ == '__main__':
    # Definizione e analisi degli argomenti da riga di comando
    parser = argparse.ArgumentParser(description='Descrizione del mio programma')
    parser.add_argument('--data', type=str)
    parser.add_argument('--retrain', type=bool)
    parser.add_argument('--mae_train', type=str)
    parser.add_argument('--mae_test', type=str)
    parser.add_argument('--params', type=str)
    parser.add_argument('--model', type=str)

    args = parser.parse_args()

    # Creazione della directory in cui verrà creato il file di output (la directory potrebbe o potrebbe non esistere).
    Path(args.mae_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.mae_test).parent.mkdir(parents=True, exist_ok=True)
    Path(args.params).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)

    _xgboost_regressor(args)