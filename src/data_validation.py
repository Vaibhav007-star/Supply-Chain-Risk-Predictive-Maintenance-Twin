"""
Data validation module for Supply Chain Risk & Predictive Maintenance Twin.
Validates raw and processed schemas, bounds, null values, and operational consistency.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class DataValidator:
    """
    Validates machine telemetry and supply chain datasets.
    """

    @staticmethod
    def validate_maintenance_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates the AI4I 2020 predictive maintenance dataset.
        """
        report: Dict[str, Any] = {
            "total_rows": len(df),
            "missing_values": int(df.isnull().sum().sum()),
            "issues": [],
            "status": "PASS"
        }

        required_cols = [
            'Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure'
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            report["issues"].append(f"Missing required columns: {missing_cols}")
            report["status"] = "FAIL"
            return False, report

        # Physical boundary checks
        if (df['Air temperature [K]'] <= 250).any() or (df['Air temperature [K]'] >= 350).any():
            report["issues"].append("Air temperature out of physical bounds (250K - 350K)")
        if (df['Process temperature [K]'] <= 250).any() or (df['Process temperature [K]'] >= 360).any():
            report["issues"].append("Process temperature out of physical bounds (250K - 360K)")
        if (df['Rotational speed [rpm]'] <= 0).any() or (df['Rotational speed [rpm]'] > 10000).any():
            report["issues"].append("Rotational speed non-positive or unrealistically high (>10k rpm)")
        if (df['Torque [Nm]'] < 0).any():
            report["issues"].append("Negative torque detected")
        if (df['Tool wear [min]'] < 0).any():
            report["issues"].append("Negative tool wear detected")

        # Label consistency
        failure_rate = float(df['Machine failure'].mean())
        report["failure_rate_percent"] = round(failure_rate * 100, 2)
        report["failure_count"] = int(df['Machine failure'].sum())

        if report["issues"]:
            report["status"] = "WARN" if report["status"] != "FAIL" else "FAIL"

        return report["status"] != "FAIL", report

    @staticmethod
    def validate_supply_chain_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates the DataCo supply chain dataset.
        """
        report: Dict[str, Any] = {
            "total_rows": len(df),
            "issues": [],
            "status": "PASS"
        }

        key_cols = [
            'Days for shipping (real)', 'Days for shipment (scheduled)',
            'Late_delivery_risk', 'Shipping Mode', 'order date (DateOrders)'
        ]
        missing_cols = [c for c in key_cols if c not in df.columns]
        if missing_cols:
            report["issues"].append(f"Missing required columns: {missing_cols}")
            report["status"] = "FAIL"
            return False, report

        # Check negative shipping days
        if (df['Days for shipping (real)'] < 0).any():
            report["issues"].append("Negative real shipping days found")
        if (df['Days for shipment (scheduled)'] < 0).any():
            report["issues"].append("Negative scheduled shipping days found")

        late_risk_mean = float(df['Late_delivery_risk'].mean())
        report["late_delivery_risk_pct"] = round(late_risk_mean * 100, 2)

        return report["status"] != "FAIL", report
