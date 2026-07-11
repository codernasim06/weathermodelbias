import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(filepath):
    """Loads the preprocessed dataset and ensures the DATE index is set."""
    return pd.read_csv(filepath, parse_dates=['DATE'], index_col='DATE')

def temporal_split(df, split_year=2015):
    """
    Splits data chronologically to mimic operational forecasting
    and prevent time-series data leakage.
    """
    train = df[df.index.year < split_year]
    test = df[df.index.year >= split_year]
    return train, test

def train_and_evaluate_models(target_name, df_processed, output_dir):
    """
    Trains and benchmarks six distinct regressors, saving performance metrics
    and model artifacts for downstream TreeSHAP auditing.
    """
    print(f"\n" + "="*60)
    print(f"STARTING TRAINING PIPELINE FOR TARGET: {target_name}")
    print("="*60 + "\n")
    
    # 1. Temporal Split
    train, test = temporal_split(df_processed, split_year=2015)
    
    # 2. Separate Features and Target
    targets_to_drop = ['T2M', 'PRECTOTCORR']
    drop_cols = targets_to_drop + ['SEASON'] 
    
    X_train = train.drop(columns=drop_cols)
    y_train = train[target_name]
    X_test = test.drop(columns=drop_cols)
    y_test = test[target_name]
    
    # 3. Feature Scaling (Crucial for Linear Regression consistency)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Save the scaler artifact
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, f'scaler_{target_name}.pkl'))
    
    # 4. Initialize the Expanded Tree-Centric Model Suite
    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=100, learning_rate=0.05, random_state=42, verbose=0)
    }
    
    metrics_summary = []
    
    # 5. Train, Predict, and Save Artifacts
    for model_name, model in models.items():
        print(f"-> Training {model_name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Generate Predictions
        predictions = model.predict(X_test_scaled)
        
        # Calculate standard IEEE performance metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics_summary.append({
            'Model': model_name,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4)
        })
        
        # Serialize model artifact for prompt loading in shap_diagnostics.py
        model_filepath = os.path.join(output_dir, f'{model_name}_{target_name}.pkl')
        joblib.dump(model, model_filepath)
        
    # 6. Construct and Print Results Table
    results_df = pd.DataFrame(metrics_summary)
    print(f"\n[RESULTS] Baseline Performance Summary for {target_name}:")
    print(results_df.to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    PROCESSED_DATA_PATH = '../data/processed/df_processed.csv'
    MODEL_OUTPUT_DIR = '../models/'
    
    try:
        # Load the long-format, interpolated data from our preprocessing script
        df = load_data(PROCESSED_DATA_PATH)
        
        # Run pipelines independently for both targets outlined in your abstract
        t2m_results = train_and_evaluate_models('T2M', df, MODEL_OUTPUT_DIR)
        precip_results = train_and_evaluate_models('PRECTOTCORR', df, MODEL_OUTPUT_DIR)
        
        print(f"\nAll models and scaling pipelines successfully archived in '{MODEL_OUTPUT_DIR}'.")
        
    except FileNotFoundError:
        print(f"Error: Could not locate '{PROCESSED_DATA_PATH}'. Ensure data_preprocessing.py has run successfully.")