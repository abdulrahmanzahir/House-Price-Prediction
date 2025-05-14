import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─── Paths ─────────────────────────────────────────────────────────────────────
# BASE_DIR: project root one level up from src/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_CSV = os.path.join(BASE_DIR, 'data', 'AmesHousing.csv')
PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'ames_processed.csv')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw Ames Housing data:
    - Drops columns with too many missing values or not useful for modeling
    - Imputes numerical columns (LotFrontage: median, others: 0)
    - Fills categorical nulls with 'None'
    """
    cols_to_drop = [
        'Id', 'Order', 'PID', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'GarageYrBlt'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Impute LotFrontage by median
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    # Fill remaining numeric NAs with 0
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(0)
    # Fill categorical NAs with 'None'
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('None')
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds new features:
    - TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
    - HouseAge = YrSold - YearBuilt
    - RemodAge = YrSold - YearRemodAdd
    """
    # Total square footage
    df['TotalSF'] = (
        df.get('TotalBsmtSF', 0)
        + df.get('1stFlrSF', 0)
        + df.get('2ndFlrSF', 0)
    )
    # Age features
    if all(col in df.columns for col in ['YrSold', 'YearBuilt']):
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    if all(col in df.columns for col in ['YrSold', 'YearRemodAdd']):
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final processing:
    - One-hot encode categorical vars
    - Standard-scale numeric features (excluding target 'SalePrice')
    """
    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)
    # Scale numeric columns except SalePrice
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'SalePrice' in num_cols:
        num_cols.remove('SalePrice')
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def main():
    # 1. Load raw data
    df = pd.read_csv(RAW_CSV)
    # 2. Clean
    df = clean_data(df)
    # 3. Feature engineering
    df = feature_engineer(df)
    # 4. Preprocess
    df = preprocess_data(df)
    # 5. Save processed data
    os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"✅ Processed data saved to {PROCESSED_CSV}")


if __name__ == '__main__':
    main()