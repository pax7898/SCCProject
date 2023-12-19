import json

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib


def _linear_regression(args):
    # Open and reads file "data"
    with open(args.data) as data_file:
        data = json.load(data_file)

    # The excted data type is 'dict', however since the file
    # was loaded as a json object, it is first loaded as a string
    # thus we need to load again from such string in order to get 
    # the dict-type object.
    data = json.loads(data)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # Initialize the model
    model = LinearRegression()

    # Parameter grid for Linear Regression
    model_param_grid = {
        'fit_intercept': [True, False],
        'positive': [True, False],
        'copy_X': [True, False],
        'n_jobs': [None, -1],  # -1 utilizza tutti i processori disponibili
    }

    # Perform GridSearch
    grid = GridSearchCV(model, model_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    best_params = str(grid.best_params_)
    best_mae = grid.best_score_

    # Save MAE of the best model in a local file
    with open(args.mae, 'w') as mae_file:
        mae_file.write(str(best_mae))

    # Save the parameters of the best model in a local file
    with open(args.params, 'w') as params_file:
        params_file.write(str(best_params))


if __name__ == '__main__':
    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--data', type=str)
    parser.add_argument('--mae', type=str)
    parser.add_argument('--params', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.mae).parent.mkdir(parents=True, exist_ok=True)
    Path(args.params).parent.mkdir(parents=True, exist_ok=True)

    _linear_regression(args)
