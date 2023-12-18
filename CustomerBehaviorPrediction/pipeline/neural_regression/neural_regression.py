import json
import argparse
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
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

    # Inizializza e addestra il modello di regressione neurale (MLPRegressor)
    model = MLPRegressor()
    model.fit(x_train, y_train)

    # Ottieni le previsioni
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)

    # Salva l'output nel file
    with open(args.mae, 'w') as mae_file:
        mae_file.write(str(mae))

if __name__ == '__main__':
    # Definizione e analisi degli argomenti da riga di comando
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--data', type=str)
    parser.add_argument('--mae', type=str)

    args = parser.parse_args()

    # Creazione della directory in cui verrà creato il file di output (la directory potrebbe o potrebbe non esistere).
    Path(args.mae).parent.mkdir(parents=True, exist_ok=True)

    _neural_regression(args)
