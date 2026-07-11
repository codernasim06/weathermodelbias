IEEE COMPAS '26/
│
├── data/
│   ├── raw/                 # Store the original 'data.csv' here 
│   └── processed/           # Save the output of reshaping script (e.g., 'df_processed.csv')
│
├── notebooks/               # For exploratory data analysis (EDA) and rapid prototyping
│   ├── 01_data_exploration.ipynb
│   └── 02_model_prototyping.ipynb
│
├── src/                     # The core, reproducible Python scripts
│   ├── __init__.py
│   ├── data_preprocessing.py # The reshaping and lag-feature script  
│   ├── model_training.py     # Code to train RF, SVM, KNN, and XGBoost baselines
│   ├── shap_diagnostics.py   # Code for extracting and analyzing SHAP values
│   └── utils.py              # Helper functions (e.g., scoring metrics, standard scalers)
│
├── figures/                 # High-resolution, IEEE-formatted outputs for the manuscript
│   ├── exploratory/         # Basic distribution plots
│   └── paper_ready/         # SHAP summary plots, seasonal dependence plots (300 DPI)
│
├── paper/                   # Manuscript files
│   ├── IEEE_template/       # The official IEEE conference LaTeX 
│   └── Abstract.txt         # Abstract
│
├── .gitignore               # To prevent uploading massive data files or local environment configs
├── requirements.txt         # List of dependencies (pandas, scikit-learn, shap, xgboost, etc.)
├── ARCHITECTURE.md          # Total root folder and files structure
└── README.md                # Project overview, setup instructions, and methodology summary