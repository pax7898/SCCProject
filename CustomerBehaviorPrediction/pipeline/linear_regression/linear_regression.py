import json
import argparse
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from joblib import dump, load


def _linear_regression(args):
    # Open and reads file "data"
    with open(args.data) as data_file:
        data = json.load(data_file)

    # Open and reads flag "retrain"
    with open(args.retrain, 'r') as file:
        retrain = file.read()

    # Load of the data dict from the json string
    data = json.loads(data)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    if retrain:
        # Initialize the model for the first training
        model = LinearRegression()
    else:
        # Load the previous model for the retraining
        model = load("linear_regression.joblib")

    # Parameter grid for Linear Regression
    model_param_grid = {
        'fit_intercept': [True, False],
        'positive': [True, False],
        'copy_X': [True, False],
        'n_jobs': [None, -1],
    }

    # GridSearch Training
    grid = GridSearchCV(model, model_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    best_params = str(grid.best_params_)

    # Mean Absolute Error Validation
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
    dump(best_model, args.model)


if __name__ == '__main__':
    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--data', type=str)
    parser.add_argument('--retrain', type=str)
    parser.add_argument('--mae_train', type=str)
    parser.add_argument('--mae_test', type=str)
    parser.add_argument('--params', type=str)
    parser.add_argument('--model', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created
    # (the directory may or may not exist).
    Path(args.mae_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.mae_test).parent.mkdir(parents=True, exist_ok=True)
    Path(args.params).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)

    _linear_regression(args)
