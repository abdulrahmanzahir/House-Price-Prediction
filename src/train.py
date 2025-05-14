import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'ames_processed.csv')


def load_data():
    df = pd.read_csv(PROCESSED_CSV)
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    max_err = max_error(y_test, preds)
    return rmse, r2, mse, mae, max_err


def main():
    mlflow.set_experiment('TRAINING')
    X_train, X_test, y_train, y_test = load_data()

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    best_rmse = float('inf')
    best_run_id = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param('model_type', name)

            # Log additional hyperparameters
            if hasattr(model, 'max_depth'):
                mlflow.log_param('max_depth', model.max_depth)
            if hasattr(model, 'min_samples_split'):
                mlflow.log_param('min_samples_split', model.min_samples_split)

            model.fit(X_train, y_train)
            rmse, r2, mse, mae, max_err = evaluate(model, X_test, y_test)

            # Log metrics
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('r2', r2)
            mlflow.log_metric('mse', mse)
            mlflow.log_metric('mae', mae)
            mlflow.log_metric('max_error', max_err)

            mlflow.sklearn.log_model(model, artifact_path='model')

            if rmse < best_rmse:
                best_rmse = rmse
                best_run_id = mlflow.active_run().info.run_id

    print(f"🏆 Best RMSE: {best_rmse} from run {best_run_id}")
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, 'Ames Housing Model')


if __name__ == '__main__':
    main()