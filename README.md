## Ames Housing Price Prediction — MLOps Project

This project implements a full machine learning lifecycle for the Ames Housing dataset, including data ingestion, preprocessing, model training, hyperparameter tuning, deployment, and monitoring using MLflow.

---

## Project Structure

```
├── data/ # Raw and processed data CSV files
├── notebooks/ # Jupyter notebooks (optional exploratory analysis)
├── scripts/ # Helper scripts (e.g., make_payload.py)
├── src/
│ ├── ingestion.py # Data download & extraction
│ ├── processing.py # Data preprocessing & encoding
│ ├── train.py # Baseline model training and logging
│ ├── tune.py # Hyperparameter tuning with MLflow + Hyperopt
│ ├── monitor.py # Model performance monitoring script
│ └── ... # Additional utility scripts
├── README.md # Project overview & instructions
├── requirements.txt # Python dependencies
└── .gitignore # Git ignore rules
```

---

## 🔧 Features
- Data ingestion and cleaning
- Feature engineering
- Model training and tuning
- Model monitoring
- REST API testing with test payload

## 📊 Dataset
- Source: Ames Housing dataset
- Target variable: `SalePrice`

  
## Setup Instructions

1.  Clone this repository:
    ```bash
    git clone https://github.com/abdulrahmanzahir/housing-price-lifecycle.git
    cd <REPO>
    ```
2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate       # Linux/macOS
    # .venv\Scripts\activate          # Windows PowerShell
    ```
3.  Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  Configure Kaggle API credentials to download the dataset:
    Place `kaggle.json` in the directory: `%USERPROFILE%\.kaggle\` (Windows) or `~/.kaggle/` (Linux/macOS).
    Make sure permissions allow read access.

---

## Usage

### Data Ingestion

Download and extract the dataset from Kaggle:
```bash
python src/ingestion.py
```

### Data Preprocessing

Encode categorical variables and save the processed dataset:
```bash
python src/processing.py
```

### Model Training

Train baseline models and log experiments to MLflow:
```bash
python src/train.py
```

### Hyperparameter Tuning

Optimize Random Forest hyperparameters using Hyperopt and log the best model:
```bash
python src/tune.py
```

### Model Serving

Serve the best registered model via MLflow model server:
```bash
mlflow models serve --model-uri "models:/Ames Housing Model/Production" --host 0.0.0.0 --port 1234
```

Send prediction requests via API (example with `curl`):
```bash
curl -X POST http://localhost:1234/invocations -H "Content-Type: application/json" --data-binary @payload.json
```

### Performance Monitoring

Run the monitoring script to sample new data, predict, and log metrics:
```bash
python src/monitor.py
```

---

## Metrics Logged

-   RMSE (Root Mean Squared Error)
-   MSE (Mean Squared Error)
-   MAE (Mean Absolute Error)
-   R² Score
-   Max Error

---

## Technologies & Tools

-   Python 3.10
-   Scikit-learn
-   MLflow (Tracking, Model Registry, Serving)
-   Hyperopt (Hyperparameter Optimization)
-   Pandas
-   Kaggle API

---

## 👤 Author
Abdulrahman Zahir  
[GitHub Profile](https://github.com/abdulrahmanzahir)
