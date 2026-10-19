"""
Unit and Integration Test Suite for Supply Chain Risk & Predictive Maintenance Twin.
Tests data validation, model inference, inventory projections, and scenario simulations.
"""
import pytest
import pandas as pd
import numpy as np

from config.settings import MAINTENANCE_MODEL_PATH, DELAY_MODEL_PATH, TWIN_STATE_PATH
from src.data_validation import DataValidator
from src.preprocessing import preprocess_maintenance_data, preprocess_supply_chain_data
from src.maintenance_model import PredictiveMaintenanceModel
from src.delay_prediction import SupplyChainDelayModel
from src.inventory_forecasting import InventoryForecaster
from src.scenario_simulator import ScenarioSimulator
from src.risk_scoring import CompositeRiskEngine


@pytest.fixture
def sample_telemetry():
    return {
        "machine_id": "TEST-001",
        "name": "Test Spindle Mill",
        "type": "H",
        "air_temperature_k": 300.5,
        "process_temperature_k": 310.2,
        "rotational_speed_rpm": 1450,
        "torque_nm": 45.0,
        "tool_wear_min": 195,
        "cost_per_hour_downtime_usd": 650,
        "daily_capacity_units": 200
    }


@pytest.fixture
def sample_component():
    return {
        "component_id": "CMP-TEST",
        "name": "Test Bearings",
        "supplier": "Test Supplier Corp",
        "current_stock": 150,
        "daily_burn_rate": 15,
        "safety_stock": 60,
        "reorder_point": 100,
        "unit_cost_usd": 120.0,
        "scheduled_lead_days": 5,
        "active_order_qty": 150
    }


@pytest.fixture
def sample_shipment():
    return {
        "shipment_id": "SHP-TEST-1",
        "component_id": "CMP-TEST",
        "component_name": "Test Bearings",
        "quantity": 150,
        "shipping_mode": "Standard Class",
        "origin_region": "Western Europe",
        "scheduled_days": 5,
        "weather_disruption_index": 0.20,
        "geopolitical_risk_index": 0.10
    }


# 1. DATA VALIDATION TESTS
def test_maintenance_validation_success():
    df_valid = pd.DataFrame({
        'Type': ['M', 'L'],
        'Air temperature [K]': [298.1, 298.2],
        'Process temperature [K]': [308.6, 308.7],
        'Rotational speed [rpm]': [1551, 1408],
        'Torque [Nm]': [42.8, 46.3],
        'Tool wear [min]': [0, 3],
        'Machine failure': [0, 1]
    })
    is_valid, report = DataValidator.validate_maintenance_data(df_valid)
    assert is_valid is True
    assert report['status'] == "PASS"


def test_maintenance_validation_catches_missing_columns():
    df_invalid = pd.DataFrame({'Type': ['M']})
    is_valid, report = DataValidator.validate_maintenance_data(df_invalid)
    assert is_valid is False
    assert report['status'] == "FAIL"


# 2. PREDICTIVE MAINTENANCE MODEL TESTS
def test_maintenance_model_prediction(sample_telemetry):
    model = PredictiveMaintenanceModel.load(MAINTENANCE_MODEL_PATH)
    pred = model.predict_risk(sample_telemetry)

    assert "failure_probability" in pred
    assert 0.0 <= pred["failure_probability"] <= 1.0
    assert pred["risk_level"] in ["HEALTHY", "MEDIUM", "HIGH", "CRITICAL"]
    assert len(pred["suspected_failure_modes"]) > 0


# 3. SUPPLY CHAIN DELAY MODEL TESTS
def test_delay_model_prediction(sample_shipment):
    model = SupplyChainDelayModel.load(DELAY_MODEL_PATH)
    pred = model.predict_delay_risk(sample_shipment)

    assert "late_probability" in pred
    assert 0.0 <= pred["late_probability"] <= 1.0
    assert pred["risk_tier"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert pred["predicted_delay_days"] >= 0.0
    assert pred["estimated_delivery_days"] >= sample_shipment["scheduled_days"]


# 4. INVENTORY FORECASTING TESTS
def test_inventory_forecast_burn_and_stockout(sample_component, sample_shipment):
    # Without inbound delivery: 150 current stock / 15 daily burn = 10 days of supply, reaches 0 on Day 10
    fc = InventoryForecaster.forecast_component_trajectory(
        sample_component,
        inbound_shipment=None,
        horizon_days=30
    )
    assert fc["stockout_day"] == 10
    assert fc["days_of_supply"] == 10.0
    assert (fc["trajectory"]["inventory_level"] >= 0).all()


def test_inventory_forecast_inbound_arrival(sample_component, sample_shipment):
    # Arrival on day 5 replenishes inventory
    fc = InventoryForecaster.forecast_component_trajectory(
        sample_component,
        inbound_shipment=sample_shipment,
        horizon_days=30,
        shipping_delay_days=0
    )
    traj = fc["trajectory"]
    # Check that day 5 received inflow
    assert traj.loc[traj['day_num'] == 5, 'inflow'].values[0] == 150


# 5. SCENARIO SIMULATOR TESTS
def test_scenario_simulator(sample_telemetry, sample_component, sample_shipment):
    twin_mock = {
        "machines": [{**sample_telemetry, "operating_hours": 3000, "failure_probability": 0.2, "risk_level": "MEDIUM", "component": "Test Bearings"}],
        "components": [sample_component],
        "shipments": [sample_shipment]
    }
    sim = ScenarioSimulator(twin_mock)

    # Scenario 1: Shipping delay
    s1 = sim.simulate_shipping_delay("CMP-TEST", additional_delay_days=10)
    assert s1["additional_delay_days"] == 10
    assert s1["simulated_arrival_day"] == 15

    # Scenario 2: Machine downtime
    s2 = sim.simulate_machinery_downtime("TEST-001", downtime_days=4)
    assert s2["lost_production_units"] == 4 * 200
    assert s2["estimated_downtime_cost_usd"] == 4 * 16 * 650

    # Scenario 3: Supplier disruption
    s3 = sim.simulate_supplier_disruption("CMP-TEST", supply_reduction_pct=50, extended_delay_days=5)
    assert s3["supply_reduction_pct"] == 50

    # Scenario 4: Demand surge
    s4 = sim.simulate_demand_surge(demand_change_pct=40)
    assert s4["demand_multiplier"] == 1.4

    # Scenario 5: Combined disruption
    s5 = sim.simulate_combined_disruption(additional_shipping_delay=5, machine_downtime_days=2, demand_change_pct=20, supplier_reduction_pct=25)
    assert s5["comparison"]["simulated_inventory_deficit"] >= s5["comparison"]["baseline_inventory_deficit"]


# 6. RISK SCORING AND RECOMMENDATIONS TESTS
def test_risk_scoring_and_recommendations(sample_telemetry, sample_component, sample_shipment):
    machines = [{**sample_telemetry, "risk_level": "CRITICAL", "failure_probability": 0.75, "suspected_failure_modes": ["Overstrain"]}]
    shipments = [{**sample_shipment, "risk_tier": "HIGH", "predicted_delay_days": 4.5, "composite_delay_risk": 0.65}]
    forecasts = [{
        "component_id": "CMP-TEST",
        "component_name": "Test Bearings",
        "stockout_day": 8,
        "cumulative_deficit_units": 120.0,
        "safety_violation_day": 5
    }]

    summary = CompositeRiskEngine.calculate_facility_summary(machines, shipments, forecasts)
    assert 0 <= summary["overall_facility_health"] <= 100
    assert summary["critical_machines_count"] == 1
    assert summary["at_risk_shipments_count"] == 1
    assert summary["stockout_components_count"] == 1

    recs = CompositeRiskEngine.generate_actionable_recommendations(summary)
    assert len(recs) >= 3
    for rec in recs:
        assert all(k in rec for k in ["category", "priority", "target", "finding", "action", "estimated_impact"])


# 7. BCG X PRESCRIPTIVE OPTIMIZATION TESTS
def test_prescriptive_inventory_optimization(sample_component, sample_shipment):
    from src.prescriptive_optimization import PrescriptiveOptimizer
    res = PrescriptiveOptimizer.optimize_inventory_policies(
        [sample_component], [sample_shipment], target_service_level=0.95
    )
    assert "components" in res
    assert len(res["components"]) == 1
    c_res = res["components"][0]
    assert c_res["optimized_safety_stock"] > 0
    assert c_res["optimized_eoq"] > 0
    assert "working_capital_impact_usd" in c_res


def test_prescriptive_maintenance_scheduling(sample_telemetry):
    from src.prescriptive_optimization import PrescriptiveOptimizer
    machines = [
        {**sample_telemetry, "machine_id": "M1", "failure_probability": 0.82, "risk_level": "CRITICAL"},
        {**sample_telemetry, "machine_id": "M2", "failure_probability": 0.45, "risk_level": "HIGH"},
        {**sample_telemetry, "machine_id": "M3", "failure_probability": 0.05, "risk_level": "HEALTHY"}
    ]
    sched = PrescriptiveOptimizer.optimize_maintenance_schedule(machines, available_technician_hours=8)
    assert sched["scheduled_interventions_count"] <= 2
    assert sched["allocated_technician_hours"] <= 8
    assert sched["total_unplanned_losses_avoided_usd"] > 0
    assert sched["roi_percentage"] > 0


def test_executive_roi_arbitrage(sample_telemetry, sample_component, sample_shipment):
    from src.prescriptive_optimization import PrescriptiveOptimizer
    machines = [{**sample_telemetry, "risk_level": "CRITICAL", "failure_probability": 0.75}]
    shipments = [{**sample_shipment, "risk_tier": "HIGH"}]
    forecasts = [{"component_id": "CMP-TEST", "stockout_day": 8, "cumulative_deficit_units": 100}]

    roi = PrescriptiveOptimizer.calculate_executive_roi_arbitrage(machines, shipments, forecasts)
    assert roi["total_value_at_risk_usd"] > 0
    assert roi["annualized_ebitda_impact_usd"] > 0
    assert roi["roi_multiplier"] >= 1.0


# 8. EXPLAINABLE AI (XAI) TESTS
def test_explainability_engine_machine_and_shipment(sample_telemetry, sample_shipment):
    from src.explainability import ExplainabilityEngine
    m_exp = ExplainabilityEngine.explain_machine_risk({
        **sample_telemetry,
        "failure_probability": 0.65,
        "power_kw": 9.2,
        "temp_difference_k": 7.9,
        "overstrain_factor": 11500
    })
    assert "contributions" in m_exp
    assert len(m_exp["contributions"]) > 0
    assert m_exp["final_failure_probability_pct"] == 65.0

    s_exp = ExplainabilityEngine.explain_shipment_delay({
        **sample_shipment,
        "composite_delay_risk": 0.55,
        "predicted_delay_days": 4.0
    })
    assert "contributions" in s_exp
    assert s_exp["final_delay_risk_pct"] == 55.0


# 9. RESILIENCE FRAMEWORK (TTS/TTR & HHI) TESTS
def test_resilience_analyzer_tts_ttr_and_hhi(sample_component, sample_shipment):
    from src.resilience_metrics import ResilienceAnalyzer
    res = ResilienceAnalyzer.calculate_component_resilience([sample_component], [sample_shipment])
    assert "resilience_records" in res
    rec = res["resilience_records"][0]
    assert rec["time_to_survive_days"] == 10.0  # 150 stock / 15 burn
    assert rec["time_to_recover_days"] > 0
    assert rec["resilience_gap_days"] == round(rec["time_to_survive_days"] - rec["time_to_recover_days"], 1)

    hhi = ResilienceAnalyzer.calculate_supplier_hhi([sample_component])
    assert hhi["hhi_index"] > 0
    assert "concentration_tier" in hhi

