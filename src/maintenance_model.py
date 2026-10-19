"""
Predictive Maintenance Machine Learning Module.
Trains, evaluates, and persists machine failure risk models using real industrial telemetry.
"""
from typing import Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from config.settings import (
    AI4I_PROCESSED_PATH, MAINTENANCE_MODEL_PATH, RANDOM_STATE,
    CRITICAL_FAILURE_THRESHOLD
)

FEATURE_COLUMNS = [
    'air_temperature_k',
    'process_temperature_k',
    'temp_difference_k',
    'rotational_speed_rpm',
    'torque_nm',
    'tool_wear_min',
    'power_kw',
    'overstrain_factor',
    'temp_ratio',
    'torque_speed_ratio',
    'type_H',
    'type_M',
    'type_L'
]

TARGET_COLUMN = 'machine_failure'
FAILURE_MODES = ['fail_tool_wear', 'fail_heat_dissipation', 'fail_power', 'fail_overstrain', 'fail_random']


class PredictiveMaintenanceModel:
    """
    Predicts machine failure risk and diagnoses potential failure modes.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        self.feature_names = FEATURE_COLUMNS
        self.metrics = {}
        self.feature_importance_df = pd.DataFrame()
        self.is_trained = False

    def train_and_evaluate(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Trains the classifier on telemetry data and records comprehensive metrics.
        """
        if df is None:
            if not AI4I_PROCESSED_PATH.exists():
                raise FileNotFoundError(f"Processed telemetry data not found at {AI4I_PROCESSED_PATH}")
            df = pd.read_parquet(AI4I_PROCESSED_PATH)

        X = df[self.feature_names]
        y = df[TARGET_COLUMN]

        # Chronological / ordered split to emulate industrial time-series progression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, shuffle=False
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "test_sample_count": len(y_test),
            "test_failures_actual": int(y_test.sum()),
            "test_failures_predicted": int(y_pred.sum())
        }

        # Feature importances
        importances = self.model.feature_importances_
        self.feature_importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return self.metrics

    def save(self, filepath: Path = MAINTENANCE_MODEL_PATH):
        """Saves model bundle to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance_df.to_dict(orient="records")
        }
        joblib.dump(bundle, filepath)
        print(f"Model saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath: Path = MAINTENANCE_MODEL_PATH) -> "PredictiveMaintenanceModel":
        """Loads model bundle from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Saved model not found at {filepath}")
        bundle = joblib.load(filepath)
        instance = cls()
        instance.model = bundle["model"]
        instance.feature_names = bundle["feature_names"]
        instance.metrics = bundle["metrics"]
        instance.feature_importance_df = pd.DataFrame(bundle["feature_importance"])
        instance.is_trained = True
        return instance

    def predict_risk(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates failure risk probability and pinpoint risk factors for a machine.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded before making predictions.")

        # Prepare single record input
        row = {}
        for col in self.feature_names:
            if col in telemetry:
                row[col] = telemetry[col]
            elif col == 'temp_difference_k':
                row[col] = telemetry.get('process_temperature_k', 308.0) - telemetry.get('air_temperature_k', 300.0)
            elif col == 'power_kw':
                row[col] = (telemetry.get('torque_nm', 40.0) * telemetry.get('rotational_speed_rpm', 1500) * (2 * np.pi / 60)) / 1000.0
            elif col == 'overstrain_factor':
                row[col] = telemetry.get('tool_wear_min', 0) * telemetry.get('torque_nm', 40.0)
            elif col == 'temp_ratio':
                row[col] = telemetry.get('process_temperature_k', 308.0) / (telemetry.get('air_temperature_k', 300.0) + 1e-5)
            elif col == 'torque_speed_ratio':
                row[col] = telemetry.get('torque_nm', 40.0) / (telemetry.get('rotational_speed_rpm', 1500) + 1e-5)
            elif col == 'type_H':
                row[col] = 1 if telemetry.get('type') == 'H' else 0
            elif col == 'type_M':
                row[col] = 1 if telemetry.get('type') == 'M' else 0
            elif col == 'type_L':
                row[col] = 1 if telemetry.get('type') == 'L' else 0
            else:
                row[col] = 0.0

        df_input = pd.DataFrame([row])[self.feature_names]
        prob = float(self.model.predict_proba(df_input)[0, 1])

        # Risk Classification
        if prob >= CRITICAL_FAILURE_THRESHOLD:
            status = "CRITICAL"
        elif prob >= 0.35:
            status = "HIGH"
        elif prob >= 0.15:
            status = "MEDIUM"
        else:
            status = "HEALTHY"

        # Rule-based diagnostic checks for physical failure modes
        suspected_modes = []
        tool_wear = telemetry.get('tool_wear_min', 0)
        overstrain = telemetry.get('overstrain_factor', row['overstrain_factor'])
        temp_diff = telemetry.get('temp_difference_k', row['temp_difference_k'])
        rpm = telemetry.get('rotational_speed_rpm', 1500)
        power = telemetry.get('power_kw', row['power_kw'])

        if tool_wear >= 200:
            suspected_modes.append("Tool Wear Failure (TWF: Tool wear >= 200 min)")
        if temp_diff < 8.6 and rpm < 1380:
            suspected_modes.append("Heat Dissipation Failure (HDF: Low airflow + low RPM)")
        if power < 3.5 or power > 9.0:
            suspected_modes.append("Power Failure (PWF: Mechanical power outside safe band 3.5 - 9.0 kW)")
        if overstrain > 11000:
            suspected_modes.append("Overstrain Failure (OSF: High tool wear & torque product)")

        return {
            "failure_probability": round(prob, 4),
            "risk_level": status,
            "suspected_failure_modes": suspected_modes if suspected_modes else ["None detected (Operating within normal parameters)"],
            "derived_power_kw": round(float(row['power_kw']), 2),
            "derived_overstrain": round(float(row['overstrain_factor']), 2),
            "derived_temp_diff_k": round(float(row['temp_difference_k']), 2)
        }
