import pandas as pd
import numpy as np
import os


def clean_data(df):
    """Clean a dataframe by filling numeric gaps and resetting the index."""
    cleaned_df = df.copy()

    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(cleaned_df[numeric_columns].mean())

    cleaned_df = cleaned_df.dropna().reset_index(drop=True)
    return cleaned_df


def feature_engineering(df):
    """Create simple engineered features for downstream modeling."""
    engineered_df = clean_data(df).copy()
    if 'category' in engineered_df.columns:
        engineered_df['is_category_A'] = (engineered_df['category'] == 'A').astype(int)
    return engineered_df


def process_data():
    # 1. Define strict relative paths based on the project architecture
    raw_data_path = '../data/raw/data.csv'
    processed_dir = '../data/processed/'
    processed_file_path = os.path.join(processed_dir, 'df_processed.csv')

    # Ensure the output directory exists before trying to save
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading raw data from: {raw_data_path}")
    
    # 2. Load the dataset (Make sure data.csv is actually inside the data/raw/ folder!)
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {raw_data_path}.")
        print("Please move your downloaded 'data.csv' file into the 'data/raw/' folder and try again.")
        return

    # 3. Reshape from Wide to Long Format
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    df_melted = df.melt(id_vars=['PARAMETER', 'YEAR'], value_vars=months, var_name='MONTH', value_name='VALUE')

    month_map = {m: i+1 for i, m in enumerate(months)}
    df_melted['MONTH_NUM'] = df_melted['MONTH'].map(month_map)
    df_pivot = df_melted.pivot_table(index=['YEAR', 'MONTH_NUM'], columns='PARAMETER', values='VALUE').reset_index()

    # 4. Create Time-Series Index
    df_pivot['DATE'] = pd.to_datetime(df_pivot['YEAR'].astype(str) + '-' + df_pivot['MONTH_NUM'].astype(str).str.zfill(2) + '-01')
    df_pivot = df_pivot.set_index('DATE').sort_index()
    df_pivot = df_pivot.drop(columns=['YEAR', 'MONTH_NUM'])

    # 5. Handle Missing Values
    df_pivot = df_pivot.replace(-999.0, np.nan)
    df_pivot = df_pivot.interpolate(method='time') 

    # 6. Feature Engineering: Seasonal Tags
    conditions = [
        df_pivot.index.month.isin([12, 1, 2]),
        df_pivot.index.month.isin([3, 4, 5]),
        df_pivot.index.month.isin([6, 7, 8, 9, 10]),
        df_pivot.index.month == 11
    ]
    choices = ['Dry', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']
    df_pivot['SEASON'] = np.select(conditions, choices, default='Unknown')

    # 7. Lag Features
    df_pivot['T2M_LAG_1'] = df_pivot['T2M'].shift(1)
    df_pivot['PRECTOTCORR_LAG_1'] = df_pivot['PRECTOTCORR'].shift(1)

    # 8. Drop NaNs and Save
    df_processed = df_pivot.dropna()
    
    df_processed.to_csv(processed_file_path)
    print(f"Successfully processed data and saved to: {processed_file_path}")

if __name__ == "__main__":
    process_data()