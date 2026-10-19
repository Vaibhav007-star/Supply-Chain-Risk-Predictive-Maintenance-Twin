"""
Global configuration and directory constants for Supply Chain Risk & Predictive Maintenance Twin.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# Data file paths
AI4I_RAW_PATH = RAW_DATA_DIR / "ai4i2020.csv"
DATACO_RAW_PATH = RAW_DATA_DIR / "DataCoSupplyChainDataset.csv"

AI4I_PROCESSED_PATH = PROCESSED_DATA_DIR / "maintenance_processed.parquet"
DATACO_PROCESSED_PATH = PROCESSED_DATA_DIR / "supply_chain_processed.parquet"
TWIN_STATE_PATH = PROCESSED_DATA_DIR / "twin_operational_state.json"

# Model paths
MAINTENANCE_MODEL_PATH = MODELS_DIR / "maintenance_classifier.joblib"
DELAY_MODEL_PATH = MODELS_DIR / "delay_risk_classifier.joblib"

# Random Seed for reproducibility
RANDOM_STATE = 42

# Operational Twin defaults
CRITICAL_FAILURE_THRESHOLD = 0.55
HIGH_DELAY_RISK_THRESHOLD = 0.50
MIN_SAFETY_STOCK_DAYS = 5
STANDARD_LEAD_TIME_DAYS = 7
