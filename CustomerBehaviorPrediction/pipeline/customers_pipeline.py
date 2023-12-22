import kfp
from kfp import dsl
from kfp.components import func_to_container_op


@func_to_container_op
def show_results(lr_mae_train: float, lr_mae_test: float, lr_params: str,
                 xgb_mae_train: float, xgb_mae_test: float, xgb_params: str,
                 neural_mae_train: float, neural_mae_test: float, neural_params: str) -> None:
    # Given the outputs from the three models components the results are shown.

    print(f"Linear Regression (mean absolute error) on Training Set: {lr_mae_train}")
    print(f"Linear Regression (mean absolute error) on Test Set: {lr_mae_test}")
    print(f"Linear Regression Best Parameters: {lr_params}\n")

    print(f"XGBoost Regressor (mean absolute error)  on Training Set: {xgb_mae_train}")
    print(f"Linear Regression (mean absolute error) on Test Set: {xgb_mae_test}")
    print(f"XGBoost Regressor Best Parameters: {xgb_params}\n")

    print(f"Neural Network Regression (mean absolute error)  on Training Set: {neural_mae_train}")
    print(f"Linear Regression (mean absolute error) on Test Set: {neural_mae_test}")
    print(f"Neural Network Regression Best Parameters: {neural_params}\n")

@func_to_container_op
def evaluate_best_model(lr:  float,
                        xgb: float,
                        neural: float) -> None:

    models = [('Linear Regressor', lr), ('XGBoost Regressor', xgb), ('Neural Regression', neural)]

    best_model = min(models, key=lambda x: x[1])

    print("Best model: " + str(best_model[0]) + "\nMae on Test Set: " + str(best_model[1]))


@dsl.pipeline(name='Customer Pipeline', description='Applies Linear Regression for Customer Spends predicition.')
def customers_pipeline():

    # Loads the yaml manifest for each component
    load = kfp.components.load_component_from_file('load_data/load_data.yaml')
    preprocess = kfp.components.load_component_from_file('preprocess_data/preprocess_data.yaml')
    linear_regression = kfp.components.load_component_from_file('linear_regression/linear_regression.yaml')
    xgboost_regressor = kfp.components.load_component_from_file('xgboost_regressor/xgboost_regressor.yaml')
    neural_regression = kfp.components.load_component_from_file('neural_regression/neural_regression.yaml')

    # Run load_data task
    load_task = load('/pipeline/dataset1.csv')
    # Run preprocess_data task
    preprocess_task = preprocess(load_task.output)

    # Run tasks "linear_regression" given
    # the output generated by "load_task".
    linear_regression_task = linear_regression(preprocess_task.output)
    xgboost_regressor_task = xgboost_regressor(preprocess_task.output)
    neural_regression_task = neural_regression(preprocess_task.output)

    # Given the outputs from "linear_regression"
    # the component "show_results" is called to print the results.
    show_results(linear_regression_task.outputs['Mae_Train'], linear_regression_task.outputs['Mae_Test'], linear_regression_task.outputs['Params'],
                 xgboost_regressor_task.outputs['Mae_Train'], xgboost_regressor_task.outputs['Mae_Test'], xgboost_regressor_task.outputs['Params'],
                 neural_regression_task.outputs['Mae_Train'], neural_regression_task.outputs['Mae_Test'], neural_regression_task.outputs['Params'])

    evaluate_best_model(linear_regression_task.outputs['Mae_Test'],
                        xgboost_regressor_task.outputs['Mae_Test'],
                        neural_regression_task.outputs['Mae_Test'])


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(customers_pipeline, 'customers_pipeline.yaml')