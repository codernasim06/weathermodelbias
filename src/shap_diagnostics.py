import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Configure matplotlib for publication-grade, crisp figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def load_data_and_artifacts(target_name, model_name, data_path, model_dir):
    """Loads preprocessed data, trained models, and corresponding scalers."""
    df = pd.read_csv(data_path, parse_dates=['DATE'], index_col='DATE')
    
    # Re-create the exact chronological test split (2015-2025)
    train = df[df.index.year < 2015]
    test = df[df.index.year >= 2015]
    
    drop_cols = ['T2M', 'PRECTOTCORR', 'SEASON']
    X_test = test.drop(columns=drop_cols)
    
    # Load scaling pipeline
    scaler = joblib.load(os.path.join(model_dir, f'scaler_{target_name}.pkl'))
    
    # Transform test features
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Load model artifact
    model = joblib.load(os.path.join(model_dir, f'{model_name}_{target_name}.pkl'))
    
    return X_test_scaled, test['SEASON'], model

def run_shap_analysis(target_name, model_name, data_path='../data/processed/df_processed.csv', model_dir='../models/', output_fig_dir='../figures/paper_ready/'):
    """
    Computes exact TreeSHAP or LinearSHAP values and exports high-resolution plots.
    Bypasses SHAP's buggy internal scatter plotting by drawing directly with Matplotlib.
    """
    print(f"\n--- Running SHAP Audit for {model_name} predicting {target_name} ---")
    os.makedirs(output_fig_dir, exist_ok=True)
    
    # 1. Load data and models
    try:
        X_test_scaled, seasons, model = load_data_and_artifacts(target_name, model_name, data_path, model_dir)
    except FileNotFoundError as e:
        print(f"Skipping {model_name} for {target_name}: Artifact not found. ({e})")
        return
    
    # 2. Compute SHAP values safely based on model class
    tree_models = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'DecisionTree']
    
    if model_name in tree_models:
        explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
        shap_values = explainer(X_test_scaled)
    else:
        explainer = shap.Explainer(model, X_test_scaled)
        shap_values = explainer(X_test_scaled)
    
    # 3. Plot 1: Global Beeswarm Plot (Works perfectly out of the box)
    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.title(f"Global Feature Impact (Beeswarm): {model_name} [{target_name}]", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    
    beeswarm_path = os.path.join(output_fig_dir, f'{model_name}_{target_name}_beeswarm.png')
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {beeswarm_path}")
    
    # 4. Plot 2: Diagnostic Seasonal Dependence Plot (Bypassing SHAP bug via direct Matplotlib)
    feature_to_plot = 'QV2M' if 'QV2M' in X_test_scaled.columns else X_test_scaled.columns[0]
    
    # Safely extract arrays from SHAP Explanation object or array fallback
    if hasattr(shap_values, "values") and hasattr(shap_values, "data"):
        # It's an Explanation object, extract using column labels
        shaps_for_feature = shap_values[:, feature_to_plot].values
        features_for_plot = shap_values[:, feature_to_plot].data
    else:
        # Fallback array slicing
        feature_idx = X_test_scaled.columns.get_loc(feature_to_plot)
        shaps_for_feature = shap_values[:, feature_idx]
        features_for_plot = X_test_scaled[feature_to_plot].values

    # Publication-grade color palette
    color_palette = {
        'Dry': '#1f77b4',          # Deep Blue
        'Pre-Monsoon': '#ff7f0e',  # Bright Orange
        'Monsoon': '#2ca02c',      # Forest Green
        'Post-Monsoon': '#d62728'  # Crimson Red
    }
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Plot each season explicitly to build a clean legend and entirely bypass the SHAP bug
    seasons_arr = seasons.values
    for season_name, color in color_palette.items():
        mask = (seasons_arr == season_name)
        if np.any(mask):
            ax.scatter(
                features_for_plot[mask], 
                shaps_for_feature[mask], 
                color=color, 
                label=season_name, 
                alpha=0.75, 
                edgecolors='none', 
                s=35
            )
            
    # Add an equilibrium reference line at SHAP = 0
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Fine-tune typography and layout
    ax.set_title(f"Diagnostic Bias Lens ({feature_to_plot}): {model_name} [{target_name}]", fontsize=11, fontweight='bold', pad=15)
    ax.set_xlabel(f"Scaled {feature_to_plot} Feature Value", fontsize=10)
    ax.set_ylabel("SHAP Value (Impact on Prediction)", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(title="South Asian Seasons", loc='best', frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    
    dependence_path = os.path.join(output_fig_dir, f'{model_name}_{target_name}_seasonal_dependence.png')
    plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dependence_path}")

if __name__ == "__main__":
    # Audit top-performing baseline architectures across both climate variables
    run_shap_analysis(target_name='T2M', model_name='LinearRegression')
    run_shap_analysis(target_name='PRECTOTCORR', model_name='RandomForest')
    run_shap_analysis(target_name='PRECTOTCORR', model_name='XGBoost')
    run_shap_analysis(target_name='PRECTOTCORR', model_name='LightGBM')
    
    print("\nSHAP visual diagnostic suites compiled successfully.")