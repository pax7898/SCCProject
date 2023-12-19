import json
import argparse
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def _neural_regression(args):
    # Apri e leggi il file "data"
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

    # Inizializza il modello di regressione neurale (MLPRegressor)
    model = MLPRegressor(max_iter=500)

    # Parametri per la grid search
    model_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }

    # Esegui la grid search
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
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--data', type=str)
    parser.add_argument('--mae', type=str)
    parser.add_argument('--params', type=str)

    args = parser.parse_args()

    # Creazione della directory in cui verranno creati i file di output (la directory potrebbe o potrebbe non esistere).
    Path(args.mae).parent.mkdir(parents=True, exist_ok=True)
    Path(args.params).parent.mkdir(parents=True, exist_ok=True)

    _neural_regression(args)
