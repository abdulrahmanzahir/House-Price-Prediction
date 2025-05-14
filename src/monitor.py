import os
import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score

# ─── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_CSV   = os.path.join(BASE_DIR, 'data', 'ames_processed.csv')
MONITOR_EXP     = 'MONITORING'
MODEL_NAME      = 'Ames Housing Model'
MODEL_VERSION   = 1
SAMPLE_FRACTION = 0.1


def main():
    # Load and sample new data
    df = pd.read_csv(PROCESSED_CSV)
    sample_df = df.sample(frac=SAMPLE_FRACTION, random_state=42)
    X_new = sample_df.drop(columns=['SalePrice'], errors='ignore')
    y_true = sample_df['SalePrice']

    # Load specific model version
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

    # Predict and compute metrics
    y_pred = model.predict(X_new)

    # Compute all metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Log monitoring metrics
    mlflow.set_experiment(MONITOR_EXP)
    with mlflow.start_run():
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('max_error', max_err)
        mlflow.log_metric('r2', r2)
        mlflow.log_param('sample_size', len(X_new))
        mlflow.log_param('model_version', MODEL_VERSION)

    print(f"Logged monitoring metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, MSE={mse:.2f}, Max Error={max_err:.2f}, R²={r2:.4f} on {len(X_new)} samples.")


if __name__ == '__main__':
    main()