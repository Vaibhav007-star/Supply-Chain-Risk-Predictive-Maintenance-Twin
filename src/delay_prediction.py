"""
Supply Chain Delay & Disruption Prediction Module.
Predicts shipment delay risks, expected delay days, and route vulnerability using DataCo data.
"""
from typing import Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error, r2_score
)

from config.settings import (
    DATACO_PROCESSED_PATH, DELAY_MODEL_PATH, RANDOM_STATE,
    HIGH_DELAY_RISK_THRESHOLD
)

SC_FEATURE_COLUMNS = [
    'shipping_days_scheduled',
    'shipping_mode',
    'order_region',
    'order_item_quantity',
    'order_item_total',
    'order_month',
    'order_dayofweek',
    'order_quarter'
]


class SupplyChainDelayModel:
    """
    Predicts late delivery probability and expected delay duration.
    """

    def __init__(self):
        # HistGradientBoosting supports categorical features natively
        self.clf = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
            categorical_features=[1, 2]  # shipping_mode, order_region
        )
        self.reg = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
            categorical_features=[1, 2]
        )
        self.feature_names = SC_FEATURE_COLUMNS
        self.metrics = {}
        self.is_trained = False
        self.category_mappings = {}

    def train_and_evaluate(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Trains delay classifier and delay duration regressor.
        """
        if df is None:
            if not DATACO_PROCESSED_PATH.exists():
                raise FileNotFoundError(f"Processed supply chain data not found at {DATACO_PROCESSED_PATH}")
            df = pd.read_parquet(DATACO_PROCESSED_PATH)

        # Ensure strings are categorical codes
        df_work = df[self.feature_names + ['late_delivery_risk', 'delay_days']].copy()
        
        # Build category mappings
        for cat_col in ['shipping_mode', 'order_region']:
            df_work[cat_col] = df_work[cat_col].astype('category')
            self.category_mappings[cat_col] = list(df_work[cat_col].cat.categories)
            df_work[cat_col] = df_work[cat_col].cat.codes

        X = df_work[self.feature_names]
        y_cls = df_work['late_delivery_risk']
        y_reg = df_work['delay_days']

        # Train/test split (80/20)
        X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
            X, y_cls, y_reg, test_size=0.20, random_state=RANDOM_STATE
        )

        # Fit classifier
        self.clf.fit(X_train, y_cls_train)
        y_cls_pred = self.clf.predict(X_test)
        y_cls_prob = self.clf.predict_proba(X_test)[:, 1]

        # Fit regressor for delay days
        self.reg.fit(X_train, y_reg_train)
        y_reg_pred = self.reg.predict(X_test)

        self.metrics = {
            "classification": {
                "accuracy": float(accuracy_score(y_cls_test, y_cls_pred)),
                "precision": float(precision_score(y_cls_test, y_cls_pred, zero_division=0)),
                "recall": float(recall_score(y_cls_test, y_cls_pred, zero_division=0)),
                "f1_score": float(f1_score(y_cls_test, y_cls_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_cls_test, y_cls_prob)),
                "confusion_matrix": confusion_matrix(y_cls_test, y_cls_pred).tolist(),
                "test_samples": len(y_cls_test)
            },
            "regression": {
                "mae_days": float(mean_absolute_error(y_reg_test, y_reg_pred)),
                "r2": float(r2_score(y_reg_test, y_reg_pred))
            }
        }
        self.is_trained = True
        return self.metrics

    def save(self, filepath: Path = DELAY_MODEL_PATH):
        """Saves model bundle to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "classifier": self.clf,
            "regressor": self.reg,
            "feature_names": self.feature_names,
            "category_mappings": self.category_mappings,
            "metrics": self.metrics
        }
        joblib.dump(bundle, filepath)
        print(f"Supply chain delay model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path = DELAY_MODEL_PATH) -> "SupplyChainDelayModel":
        """Loads model bundle from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model bundle not found at {filepath}")
        bundle = joblib.load(filepath)
        instance = cls()
        instance.clf = bundle["classifier"]
        instance.reg = bundle["regressor"]
        instance.feature_names = bundle["feature_names"]
        instance.category_mappings = bundle["category_mappings"]
        instance.metrics = bundle["metrics"]
        instance.is_trained = True
        return instance

    def predict_delay_risk(self, shipment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates delay probability, expected delay duration, and risk tier for a shipment.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded before making predictions.")

        # Encode categorical features
        def get_cat_code(col: str, val: str) -> int:
            cats = self.category_mappings.get(col, [])
            if val in cats:
                return cats.index(val)
            return 0

        mode_code = get_cat_code('shipping_mode', shipment.get('shipping_mode', 'Standard Class'))
        region_code = get_cat_code('order_region', shipment.get('origin_region', 'Western Europe'))

        row = {
            'shipping_days_scheduled': shipment.get('scheduled_days', 4),
            'shipping_mode': mode_code,
            'order_region': region_code,
            'order_item_quantity': shipment.get('quantity', 100),
            'order_item_total': shipment.get('quantity', 100) * 50.0,
            'order_month': shipment.get('order_month', 10),
            'order_dayofweek': shipment.get('order_dayofweek', 2),
            'order_quarter': shipment.get('order_quarter', 4)
        }

        df_in = pd.DataFrame([row])[self.feature_names]
        prob = float(self.clf.predict_proba(df_in)[0, 1])
        expected_delay_days = max(0.0, float(self.reg.predict(df_in)[0]))

        # Include external risk modifiers if present (weather, geopolitical)
        weather_idx = shipment.get('weather_disruption_index', 0.0)
        geopolitical_idx = shipment.get('geopolitical_risk_index', 0.0)

        # Multi-factor calibrated risk index
        composite_risk_score = min(1.0, prob * 0.7 + weather_idx * 0.2 + geopolitical_idx * 0.1)

        if composite_risk_score >= 0.70:
            status = "CRITICAL"
        elif composite_risk_score >= HIGH_DELAY_RISK_THRESHOLD:
            status = "HIGH"
        elif composite_risk_score >= 0.30:
            status = "MEDIUM"
        else:
            status = "LOW"

        additional_delay = expected_delay_days + (weather_idx * 4.0) + (geopolitical_idx * 3.0)

        return {
            "late_probability": round(prob, 4),
            "composite_delay_risk": round(composite_risk_score, 4),
            "risk_tier": status,
            "predicted_delay_days": round(additional_delay, 1),
            "estimated_delivery_days": round(shipment.get('scheduled_days', 4) + additional_delay, 1),
            "weather_impact_days": round(weather_idx * 4.0, 1),
            "geopolitical_impact_days": round(geopolitical_idx * 3.0, 1)
        }
