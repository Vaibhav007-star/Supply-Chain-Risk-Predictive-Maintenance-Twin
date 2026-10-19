"""
Data loader and digital twin state generator.
Loads, validates, preprocesses, and caches telemetry and supply chain records.
"""
import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

from config.settings import (
    AI4I_RAW_PATH, DATACO_RAW_PATH,
    AI4I_PROCESSED_PATH, DATACO_PROCESSED_PATH, TWIN_STATE_PATH,
    RANDOM_STATE
)
from src.data_validation import DataValidator
from src.preprocessing import preprocess_maintenance_data, preprocess_supply_chain_data


def load_raw_maintenance_data() -> pd.DataFrame:
    """Loads raw AI4I 2020 CSV dataset."""
    if not AI4I_RAW_PATH.exists():
        raise FileNotFoundError(f"Raw maintenance dataset not found at {AI4I_RAW_PATH}")
    return pd.read_csv(AI4I_RAW_PATH)


def load_raw_supply_chain_data(nrows: int = None) -> pd.DataFrame:
    """Loads raw DataCo CSV dataset."""
    if not DATACO_RAW_PATH.exists():
        raise FileNotFoundError(f"Raw supply chain dataset not found at {DATACO_RAW_PATH}")
    return pd.read_csv(DATACO_RAW_PATH, encoding='ISO-8859-1', nrows=nrows)


def prepare_and_cache_datasets(force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates, preprocesses, and saves Parquet artifacts for fast retrieval.
    """
    # 1. Maintenance data
    if force_reprocess or not AI4I_PROCESSED_PATH.exists():
        print("Loading and validating raw maintenance telemetry...")
        df_m_raw = load_raw_maintenance_data()
        valid, val_report = DataValidator.validate_maintenance_data(df_m_raw)
        print(f"Maintenance Validation: {val_report['status']} (Failure rate: {val_report.get('failure_rate_percent')}%)")
        
        df_m_clean = preprocess_maintenance_data(df_m_raw)
        AI4I_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_m_clean.to_parquet(AI4I_PROCESSED_PATH, index=False)
        print(f"Saved processed maintenance data to {AI4I_PROCESSED_PATH} ({len(df_m_clean)} rows)")
    else:
        df_m_clean = pd.read_parquet(AI4I_PROCESSED_PATH)

    # 2. Supply chain data
    if force_reprocess or not DATACO_PROCESSED_PATH.exists():
        print("Loading and validating raw supply chain data...")
        df_sc_raw = load_raw_supply_chain_data()
        valid_sc, val_report_sc = DataValidator.validate_supply_chain_data(df_sc_raw)
        print(f"Supply Chain Validation: {val_report_sc['status']} (Late Risk: {val_report_sc.get('late_delivery_risk_pct')}%)")

        df_sc_clean = preprocess_supply_chain_data(df_sc_raw)
        DATACO_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_sc_clean.to_parquet(DATACO_PROCESSED_PATH, index=False)
        print(f"Saved processed supply chain data to {DATACO_PROCESSED_PATH} ({len(df_sc_clean)} rows)")
    else:
        df_sc_clean = pd.read_parquet(DATACO_PROCESSED_PATH)

    # 3. Build Digital Twin operational mapping if not already generated
    if force_reprocess or not TWIN_STATE_PATH.exists():
        print("Building Digital Twin factory operational topology...")
        build_digital_twin_topology(df_m_clean, df_sc_clean)

    return df_m_clean, df_sc_clean


def build_digital_twin_topology(df_m: pd.DataFrame, df_sc: pd.DataFrame) -> Dict[str, Any]:
    """
    Creates an industrial digital twin state linking real machine telemetry,
    production lines, component inventory, and active inbound shipments.
    """
    np.random.seed(RANDOM_STATE)

    # Define 15 machines across 3 production bays
    machines_spec = [
        {"machine_id": "CNC-101", "bay": "Bay A (Machining)", "type": "H", "name": "Heavy 5-Axis Milling 1", "component": "Cutting Inserts"},
        {"machine_id": "CNC-102", "bay": "Bay A (Machining)", "type": "H", "name": "Heavy 5-Axis Milling 2", "component": "Cutting Inserts"},
        {"machine_id": "CNC-103", "bay": "Bay A (Machining)", "type": "M", "name": "Vertical Machining Center 1", "component": "Spindle Bearings"},
        {"machine_id": "CNC-104", "bay": "Bay A (Machining)", "type": "M", "name": "Vertical Machining Center 2", "component": "Spindle Bearings"},
        {"machine_id": "CNC-105", "bay": "Bay A (Machining)", "type": "L", "name": "High-Speed Drill & Tap 1", "component": "Drill Bits"},
        
        {"machine_id": "LAT-201", "bay": "Bay B (Turning)", "type": "H", "name": "Precision CNC Lathe 1", "component": "Carbide Tooling"},
        {"machine_id": "LAT-202", "bay": "Bay B (Turning)", "type": "M", "name": "Precision CNC Lathe 2", "component": "Carbide Tooling"},
        {"machine_id": "LAT-203", "bay": "Bay B (Turning)", "type": "M", "name": "Automatic Multi-Spindle Lathe", "component": "Spindle Bearings"},
        {"machine_id": "LAT-204", "bay": "Bay B (Turning)", "type": "L", "name": "Turret Lathe A", "component": "Drive Belts"},
        {"machine_id": "LAT-205", "bay": "Bay B (Turning)", "type": "L", "name": "Turret Lathe B", "component": "Drive Belts"},

        {"machine_id": "PRS-301", "bay": "Bay C (Stamping & Press)", "type": "H", "name": "800-Ton Hydraulic Press", "component": "Hydraulic Seals"},
        {"machine_id": "PRS-302", "bay": "Bay C (Stamping & Press)", "type": "H", "name": "500-Ton Hydraulic Press", "component": "Hydraulic Seals"},
        {"machine_id": "PRS-303", "bay": "Bay C (Stamping & Press)", "type": "M", "name": "Pneumatic Stamping Line", "component": "Pressure Valves"},
        {"machine_id": "PRS-304", "bay": "Bay C (Stamping & Press)", "type": "M", "name": "Laser Blanking Machine", "component": "Optical Lenses"},
        {"machine_id": "PRS-305", "bay": "Bay C (Stamping & Press)", "type": "L", "name": "Robotic Transfer Press", "component": "Servo Motors"}
    ]

    # Sample representative telemetry from real dataset
    telemetry_samples = df_m.sample(len(machines_spec), random_state=RANDOM_STATE).to_dict(orient="records")

    machines = []
    for spec, telemetry in zip(machines_spec, telemetry_samples):
        # Merge telemetry
        machine_record = {
            **spec,
            "air_temperature_k": round(float(telemetry['air_temperature_k']), 2),
            "process_temperature_k": round(float(telemetry['process_temperature_k']), 2),
            "rotational_speed_rpm": int(telemetry['rotational_speed_rpm']),
            "torque_nm": round(float(telemetry['torque_nm']), 2),
            "tool_wear_min": int(telemetry['tool_wear_min']),
            "temp_difference_k": round(float(telemetry['temp_difference_k']), 2),
            "power_kw": round(float(telemetry['power_kw']), 2),
            "overstrain_factor": round(float(telemetry['overstrain_factor']), 2),
            "historical_failure_label": int(telemetry['machine_failure']),
            "failure_types": {
                "TWF": int(telemetry['fail_tool_wear']),
                "HDF": int(telemetry['fail_heat_dissipation']),
                "PWF": int(telemetry['fail_power']),
                "OSF": int(telemetry['fail_overstrain']),
                "RNF": int(telemetry['fail_random'])
            },
            "operating_hours": int(np.random.randint(1200, 8500)),
            "daily_capacity_units": int(np.random.choice([150, 200, 300, 500])),
            "cost_per_hour_downtime_usd": int(np.random.choice([450, 650, 900, 1500]))
        }
        machines.append(machine_record)

    # Extract 8 critical components with realistic inventory parameters
    components = [
        {"component_id": "CMP-101", "name": "Cutting Inserts", "supplier": "Apex Precision Tooling (Germany)", "current_stock": 280, "daily_burn_rate": 35, "safety_stock": 140, "reorder_point": 250, "unit_cost_usd": 65.0, "primary_route": "Western Europe to US Midwest", "active_order_qty": 350, "scheduled_lead_days": 6},
        {"component_id": "CMP-102", "name": "Spindle Bearings", "supplier": "Koyo-Nippon Bearings (Japan)", "current_stock": 95, "daily_burn_rate": 12, "safety_stock": 60, "reorder_point": 100, "unit_cost_usd": 320.0, "primary_route": "East Asia to US Midwest", "active_order_qty": 120, "scheduled_lead_days": 10},
        {"component_id": "CMP-103", "name": "Drill Bits", "supplier": "Sandvik Coromant (Sweden)", "current_stock": 420, "daily_burn_rate": 50, "safety_stock": 200, "reorder_point": 350, "unit_cost_usd": 45.0, "primary_route": "Western Europe to US Midwest", "active_order_qty": 500, "scheduled_lead_days": 5},
        {"component_id": "CMP-104", "name": "Carbide Tooling", "supplier": "Kennametal Global (USA)", "current_stock": 180, "daily_burn_rate": 20, "safety_stock": 90, "reorder_point": 150, "unit_cost_usd": 110.0, "primary_route": "Domestic Freight (Ohio to Michigan)", "active_order_qty": 200, "scheduled_lead_days": 3},
        {"component_id": "CMP-105", "name": "Drive Belts", "supplier": "Gates Industrial (Mexico)", "current_stock": 140, "daily_burn_rate": 15, "safety_stock": 75, "reorder_point": 120, "unit_cost_usd": 85.0, "primary_route": "Central America Border Rail", "active_order_qty": 150, "scheduled_lead_days": 7},
        {"component_id": "CMP-106", "name": "Hydraulic Seals", "supplier": "Freudenberg Sealing (Germany)", "current_stock": 85, "daily_burn_rate": 14, "safety_stock": 70, "reorder_point": 110, "unit_cost_usd": 145.0, "primary_route": "Western Europe to US Midwest", "active_order_qty": 150, "scheduled_lead_days": 8},
        {"component_id": "CMP-107", "name": "Pressure Valves", "supplier": "Parker Hannifin (USA)", "current_stock": 72, "daily_burn_rate": 8, "safety_stock": 40, "reorder_point": 65, "unit_cost_usd": 280.0, "primary_route": "Domestic Air Freight", "active_order_qty": 80, "scheduled_lead_days": 2},
        {"component_id": "CMP-108", "name": "Servo Motors", "supplier": "Siemens Motion (Germany)", "current_stock": 28, "daily_burn_rate": 3, "safety_stock": 18, "reorder_point": 25, "unit_cost_usd": 1250.0, "primary_route": "Trans-Atlantic Express", "active_order_qty": 30, "scheduled_lead_days": 12}
    ]

    # Inbound shipments sampled from DataCo real shipping records
    sc_sample = df_sc.sample(len(components), random_state=RANDOM_STATE).to_dict(orient="records")
    shipments = []
    for idx, (comp, sc_row) in enumerate(zip(components, sc_sample)):
        sched_days = comp['scheduled_lead_days']
        hist_real_days = int(sc_row['shipping_days_real']) if not pd.isna(sc_row['shipping_days_real']) else sched_days
        shipment = {
            "shipment_id": f"SHP-2026-{1000 + idx}",
            "component_id": comp['component_id'],
            "component_name": comp['name'],
            "supplier": comp['supplier'],
            "quantity": comp['active_order_qty'],
            "shipping_mode": str(sc_row.get('shipping_mode', 'Standard Class')),
            "origin_region": str(sc_row.get('order_region', 'Western Europe')),
            "scheduled_days": sched_days,
            "estimated_real_days": max(sched_days, hist_real_days),
            "historical_delay_risk": int(sc_row.get('late_delivery_risk', 0)),
            "weather_disruption_index": round(float(np.random.uniform(0.05, 0.45)), 2),
            "geopolitical_risk_index": round(float(np.random.uniform(0.02, 0.35)), 2),
            "status": "In Transit"
        }
        shipments.append(shipment)

    twin_state = {
        "metadata": {
            "facility_name": "Antigravity Advanced Manufacturing Facility - Line 4",
            "total_machines": len(machines),
            "total_components": len(components),
            "total_active_shipments": len(shipments),
            "last_synced": "2026-10-19 09:10:00"
        },
        "machines": machines,
        "components": components,
        "shipments": shipments
    }

    with open(TWIN_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(twin_state, f, indent=2)

    return twin_state
