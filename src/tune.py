import os
import mlflow
import mlflow.sklearn
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'ames_processed.csv')
EXPERIMENT_NAME = 'TUNING'

def load_data():
    df = pd.read_csv(PROCESSED_CSV)
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def objective(params):
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        X_train, X_test, y_train, y_test = load_data()
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        evs = explained_variance_score(y_test, preds)
        max_err = max_error(y_test, preds)
        
        # Log metrics
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('r2', r2)
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('evs', evs)
        mlflow.log_metric('max_error', max_err)
        
        # Log model
        mlflow.sklearn.log_model(model, artifact_path='model')
        
        # Return the loss (RMSE for hyperparameter tuning)
        return {'loss': rmse, 'status': STATUS_OK}

def main():
    # Create or fetch experiment
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Define search space for hyperparameters
    space = {
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'max_depth': hp.choice('max_depth', [None, 5, 10, 20]),
        'max_features': hp.choice('max_features', ['sqrt', 'log2']),
        'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
        'bootstrap': hp.choice('bootstrap', [True, False]),
    }
    
    trials = Trials()

    # Run the optimization
    fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
    
    # Fetch the best run
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    df = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                            order_by=['metrics.rmse ASC'],
                            max_results=1)
    best_run_id = df.loc[0, 'run_id']
    print(f"🏆 Best tuned model run id: {best_run_id}")

    # Register the best model
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, 'Ames Housing Model')

if __name__ == '__main__':
    main()