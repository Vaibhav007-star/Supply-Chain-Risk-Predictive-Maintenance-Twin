"""
Explainable AI (XAI) & Root Cause Attribution Module.
Provides local feature attribution and waterfall breakdowns for:
1. Machine Telemetry Failure Risk (Physics-guided SHAP-style breakdown)
2. Inbound Shipment Delay Risk (Route, Carrier, Weather, Geopolitical attribution)
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd


class ExplainabilityEngine:
    """
    Computes local feature attributions and root-cause breakdowns
    for individual asset predictions.
    """

    @staticmethod
    def explain_machine_risk(machine: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decomposes the machine's failure probability into constituent physics drivers
        relative to normal baseline operating conditions.
        """
        prob = machine.get('failure_probability', 0.05)
        # Healthy reference baselines
        ref_rpm = 1530.0
        ref_tool_wear = 80.0
        ref_power = 6.2
        ref_temp_diff = 9.8
        ref_overstrain = 4500.0

        # Current machine metrics
        curr_rpm = machine.get('rotational_speed_rpm', 1500)
        curr_wear = machine.get('tool_wear_min', 0)
        curr_power = machine.get('power_kw', 6.0)
        curr_temp_diff = machine.get('temp_difference_k', 9.5)
        curr_overstrain = machine.get('overstrain_factor', 4000)

        # Baseline expected probability for a healthy machine
        base_prob = 0.025

        # Physics-grounded relative contributions
        wear_effect = max(0.0, (curr_wear - ref_tool_wear) / 160.0) * 0.35
        rpm_effect = max(0.0, abs(curr_rpm - ref_rpm) / 300.0) * 0.20
        power_effect = (0.25 if (curr_power < 3.8 or curr_power > 8.5) else 0.02)
        temp_effect = max(0.0, (8.8 - curr_temp_diff) / 3.0) * 0.25 if curr_temp_diff < 8.8 else -0.01
        overstrain_effect = max(0.0, (curr_overstrain - ref_overstrain) / 7000.0) * 0.30

        raw_effects = {
            "Tool Wear Degradation": wear_effect,
            "Rotational Speed Delta": rpm_effect,
            "Mechanical Power Deviation": power_effect,
            "Thermal Heat Dissipation (ΔT)": temp_effect,
            "Torque-Wear Overstrain": overstrain_effect
        }

        # Normalize total sum of effects to match (prob - base_prob)
        total_raw = sum(max(0.001, v) for v in raw_effects.values())
        target_delta = max(0.0, prob - base_prob)

        normalized_contributions = []
        cumulative = base_prob
        
        # Sort factors by magnitude
        sorted_factors = sorted(raw_effects.items(), key=lambda x: x[1], reverse=True)

        for factor_name, raw_val in sorted_factors:
            scaled_delta = (raw_val / total_raw) * target_delta if target_delta > 0 else 0.002
            normalized_contributions.append({
                "factor": factor_name,
                "delta": round(scaled_delta * 100, 2),  # percentage points
                "cumulative": round((cumulative + scaled_delta) * 100, 2)
            })
            cumulative += scaled_delta

        return {
            "machine_id": machine.get('machine_id', 'Unknown'),
            "name": machine.get('name', 'Asset'),
            "baseline_probability_pct": round(base_prob * 100, 2),
            "final_failure_probability_pct": round(prob * 100, 2),
            "top_risk_driver": sorted_factors[0][0] if sorted_factors else "Operating Normally",
            "contributions": normalized_contributions
        }

    @staticmethod
    def explain_shipment_delay(shipment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decomposes shipment late delivery risk and delay duration into root-cause factors:
        Carrier Lead Time baseline, Weather Disruption, and Geopolitical Risk.
        """
        comp_risk = shipment.get('composite_delay_risk', 0.20)
        pred_delay = shipment.get('predicted_delay_days', 0)
        weather_idx = shipment.get('weather_disruption_index', 0.1)
        geo_idx = shipment.get('geopolitical_risk_index', 0.1)
        mode = shipment.get('shipping_mode', 'Standard Class')
        origin = shipment.get('origin_region', 'Global')

        # Baseline delay probability for standard route
        base_risk = 0.15

        # Relative factor contributions
        weather_contrib = max(0.01, weather_idx * 0.45)
        geo_contrib = max(0.01, geo_idx * 0.35)
        mode_penalty = 0.15 if mode == 'Standard Class' else (0.05 if mode == 'Second Class' else 0.01)

        total_raw = weather_contrib + geo_contrib + mode_penalty
        delta = max(0.0, comp_risk - base_risk)

        contributions = [
            {
                "factor": f"Weather & Climate Volatility ({round(weather_idx*100)}%)",
                "delta_pct": round((weather_contrib / total_raw) * delta * 100, 1) if delta > 0 else 1.2
            },
            {
                "factor": f"Geopolitical & Customs Friction ({round(geo_idx*100)}%)",
                "delta_pct": round((geo_contrib / total_raw) * delta * 100, 1) if delta > 0 else 0.8
            },
            {
                "factor": f"Carrier Transit Mode ({mode})",
                "delta_pct": round((mode_penalty / total_raw) * delta * 100, 1) if delta > 0 else 0.5
            }
        ]

        # Top driver
        top_driver = max(contributions, key=lambda x: x['delta_pct'])['factor'].split(" (")[0]

        return {
            "shipment_id": shipment.get('shipment_id', 'Unknown'),
            "component_name": shipment.get('component_name', 'Component'),
            "origin_region": origin,
            "baseline_delay_risk_pct": round(base_risk * 100, 1),
            "final_delay_risk_pct": round(comp_risk * 100, 1),
            "predicted_delay_days": pred_delay,
            "top_root_cause": top_driver,
            "contributions": contributions
        }
