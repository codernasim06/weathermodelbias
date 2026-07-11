import numpy as np
import pandas as pd

def calculate_custom_metrics(y_true, y_pred, target_name):
    """
    Calculates standard metrics along with physical domain-specific insights
    (e.g., tracking extreme overestimation vs underestimation).
    """
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # Custom insight: Percentage of cases where model significantly over-predicts
    # Useful for highlighting monsoon overestimation biases in the paper
    significant_overprediction = np.sum(errors > (1.5 * mae)) / len(errors) * 100
    
    print(f"[{target_name} Evaluation Insights]")
    print(f" -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    print(f" -> Extreme Overprediction Rate: {significant_overprediction:.2f}%")
    
    return {"MAE": mae, "RMSE": rmse, "Overpred_Rate": significant_overprediction}