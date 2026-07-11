import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import clean_data, feature_engineering

# 1. Setup a fixture to provide dummy data for testing
@pytest.fixture
def sample_dataframe():
    data = {
        'id': [1, 2, 3],
        'value': [10.5, np.nan, 20.0],
        'category': ['A', 'B', 'A']
    }
    return pd.DataFrame(data)

# 2. Test Data Cleaning
def test_clean_data_handles_nans(sample_dataframe):
    df_cleaned = clean_data(sample_dataframe)
    
    # Assert that no NaNs remain
    assert df_cleaned.isnull().sum().sum() == 0, "Pipeline failed to remove all NaNs"
    
    # Assert the index was reset
    assert df_cleaned.index[0] == 0, "Index was not reset after cleaning"

# 3. Test Feature Engineering
def test_feature_engineering_output_columns(sample_dataframe):
    df_features = feature_engineering(sample_dataframe)
    
    # Define expected columns
    expected_cols = ['id', 'value', 'category', 'is_category_A']
    
    for col in expected_cols:
        assert col in df_features.columns, f"Column {col} missing from output"

# 4. Test Model Input Shapes (Crucial for ML)
def test_model_input_shape():
    # Dummy input
    dummy_input = np.zeros((10, 5)) # 10 samples, 5 features
    
    # Assume you have a function that prepares data for the model
    # assert prepare_for_model(dummy_input).shape[1] == 5
    pass