"""
FastAPI Server for BCG X Supply Chain Risk & Predictive Maintenance Digital Twin.
Serves the web application and provides high-speed RESTful API endpoints for
digital twin telemetry, machine learning inference, prescriptive optimization,
explainable AI (XAI), and resilience analytics.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import (
    TWIN_STATE_PATH, MAINTENANCE_MODEL_PATH, DELAY_MODEL_PATH
)
from src.data_loader import prepare_and_cache_datasets
from src.maintenance_model import PredictiveMaintenanceModel
from src.delay_prediction import SupplyChainDelayModel
from src.inventory_forecasting import InventoryForecaster
from src.scenario_simulator import ScenarioSimulator
from src.risk_scoring import CompositeRiskEngine
from src.prescriptive_optimization import PrescriptiveOptimizer
from src.explainability import ExplainabilityEngine
from src.resilience_metrics import ResilienceAnalyzer


import pandas as pd


def clean_for_json(obj: Any) -> Any:
    """Recursively converts pandas DataFrames, numpy numbers, and arrays into standard Python types."""
    if isinstance(obj, pd.DataFrame):
        return [clean_for_json(row) for row in obj.to_dict(orient='records')]
    elif isinstance(obj, pd.Series):
        return clean_for_json(obj.to_dict())
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [clean_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    return obj


app = FastAPI(
    title="BCG X Supply Chain & Predictive Maintenance Digital Twin API",
    version="2.0.0",
    description="Enterprise Decision Intelligence and Digital Twin Platform",
    redirect_slashes=False
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
class TwinStateContainer:
    twin_state: Dict[str, Any] = {}
    m_model: Optional[PredictiveMaintenanceModel] = None
    sc_model: Optional[SupplyChainDelayModel] = None
    comp_forecasts: List[Dict[str, Any]] = []
    facility_summary: Dict[str, Any] = {}
    simulator: Optional[ScenarioSimulator] = None

state_container = TwinStateContainer()


def initialize_twin():
    """Initializes models, twin topology, and baseline forecasts."""
    if not TWIN_STATE_PATH.exists():
        prepare_and_cache_datasets()

    with open(TWIN_STATE_PATH, "r", encoding="utf-8") as f:
        twin = json.load(f)

    m_model = PredictiveMaintenanceModel.load(MAINTENANCE_MODEL_PATH)
    sc_model = SupplyChainDelayModel.load(DELAY_MODEL_PATH)

    # Enrich machines
    for m in twin['machines']:
        pred = m_model.predict_risk(m)
        m.update(pred)

    # Enrich shipments
    for s in twin['shipments']:
        sc_pred = sc_model.predict_delay_risk(s)
        s.update(sc_pred)

    # Baseline forecasts
    forecasts = []
    for comp in twin['components']:
        shipment = next((s for s in twin['shipments'] if s['component_id'] == comp['component_id']), None)
        fc = InventoryForecaster.forecast_component_trajectory(comp, shipment)
        forecasts.append(fc)

    summary = CompositeRiskEngine.calculate_facility_summary(
        twin['machines'], twin['shipments'], forecasts
    )

    state_container.twin_state = twin
    state_container.m_model = m_model
    state_container.sc_model = sc_model
    state_container.comp_forecasts = forecasts
    state_container.facility_summary = summary
    state_container.simulator = ScenarioSimulator(twin)


# Initialize on startup
initialize_twin()


# -------------------------------------------------------------
# Request Schemas
# -------------------------------------------------------------
class InventoryOptimizeRequest(BaseModel):
    target_service_level: float = 0.95
    order_cost_usd: float = 150.0

class MaintenanceOptimizeRequest(BaseModel):
    available_technician_hours: int = 24
    cost_per_tech_hour_usd: float = 85.0

class ShippingDelaySimRequest(BaseModel):
    component_id: str
    additional_delay_days: int = 6

class MachineDowntimeSimRequest(BaseModel):
    machine_id: str
    downtime_days: int = 3

class SupplierDisruptionSimRequest(BaseModel):
    component_id: str
    supply_reduction_pct: int = 50
    extended_delay_days: int = 8

class DemandSurgeSimRequest(BaseModel):
    demand_change_pct: int = 30

class CombinedSimRequest(BaseModel):
    additional_shipping_delay: int = 6
    machine_downtime_days: int = 3
    demand_change_pct: int = 25
    supplier_reduction_pct: int = 20


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------
@app.get("/api/health")
def healthcheck():
    return clean_for_json({
        "status": "healthy",
        "service": "BCG X Digital Twin API",
        "models_loaded": {
            "predictive_maintenance": state_container.m_model is not None,
            "supply_chain_delay": state_container.sc_model is not None
        },
        "total_assets": len(state_container.twin_state.get('machines', [])),
        "total_components": len(state_container.twin_state.get('components', []))
    })


@app.get("/api/state")
def get_full_state():
    """Returns top-level executive state, facility summary, ROI arbitrage, and resilience profile."""
    twin = state_container.twin_state
    summary = state_container.facility_summary
    roi_arbitrage = PrescriptiveOptimizer.calculate_executive_roi_arbitrage(
        twin['machines'], twin['shipments'], state_container.comp_forecasts
    )
    resilience_data = ResilienceAnalyzer.calculate_component_resilience(
        twin['components'], twin['shipments']
    )
    hhi_data = ResilienceAnalyzer.calculate_supplier_hhi(twin['components'])

    return clean_for_json({
        "metadata": twin['metadata'],
        "facility_summary": summary,
        "roi_arbitrage": roi_arbitrage,
        "resilience_overview": {
            "average_network_resilience_score": resilience_data['average_network_resilience_score'],
            "vulnerable_components_count": resilience_data['vulnerable_components_count'],
            "total_components": resilience_data['total_components'],
            "hhi_index": hhi_data['hhi_index'],
            "concentration_tier": hhi_data['concentration_tier']
        }
    })


@app.get("/api/machines")
def get_machines():
    """Returns all machine assets with telemetry, failure probabilities, and status badges."""
    return clean_for_json(state_container.twin_state.get('machines', []))


@app.get("/api/machine/{machine_id}")
def get_machine_detail(machine_id: str):
    """Returns specific machine telemetry, diagnostics, and local XAI attribution."""
    m = next((item for item in state_container.twin_state['machines'] if item['machine_id'] == machine_id), None)
    if not m:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")

    xai = ExplainabilityEngine.explain_machine_risk(m)
    return clean_for_json({
        "machine": m,
        "explainability": xai
    })


@app.get("/api/shipments")
def get_shipments():
    """Returns active inbound shipments with delay risks and route indicators."""
    return clean_for_json(state_container.twin_state.get('shipments', []))


@app.get("/api/shipment/{shipment_id}")
def get_shipment_detail(shipment_id: str):
    """Returns shipment details and XAI delay root-cause attribution."""
    s = next((item for item in state_container.twin_state['shipments'] if item['shipment_id'] == shipment_id), None)
    if not s:
        raise HTTPException(status_code=404, detail=f"Shipment {shipment_id} not found")

    xai = ExplainabilityEngine.explain_shipment_delay(s)
    return clean_for_json({
        "shipment": s,
        "explainability": xai
    })


@app.get("/api/inventory")
def get_inventory():
    """Returns all components with 30-day projection trajectories and safety thresholds."""
    results = []
    for comp in state_container.twin_state['components']:
        shp = next((s for s in state_container.twin_state['shipments'] if s['component_id'] == comp['component_id']), None)
        fc = InventoryForecaster.forecast_component_trajectory(comp, shp)
        # Convert dataframe trajectory to list of dicts for JSON
        fc_dict = {k: v for k, v in fc.items() if k != 'trajectory'}
        fc_dict['trajectory'] = fc['trajectory'].to_dict(orient='records')
        fc_dict['unit_cost_usd'] = comp.get('unit_cost_usd', 100.0)
        results.append(fc_dict)
    return clean_for_json(results)


@app.post("/api/optimize/inventory")
def optimize_inventory(req: InventoryOptimizeRequest):
    """Calculates mathematically optimal Safety Stock, ROP, and EOQ."""
    res = PrescriptiveOptimizer.optimize_inventory_policies(
        state_container.twin_state['components'],
        state_container.twin_state['shipments'],
        target_service_level=req.target_service_level,
        order_cost_usd=req.order_cost_usd
    )
    return clean_for_json(res)


@app.post("/api/optimize/maintenance")
def optimize_maintenance(req: MaintenanceOptimizeRequest):
    """Resource-constrained maintenance scheduling optimization."""
    res = PrescriptiveOptimizer.optimize_maintenance_schedule(
        state_container.twin_state['machines'],
        available_technician_hours=req.available_technician_hours,
        cost_per_tech_hour_usd=req.cost_per_tech_hour_usd
    )
    return clean_for_json(res)


@app.get("/api/resilience")
def get_resilience_matrix():
    """Returns BCG Time-to-Survive (TTS) vs Time-to-Recover (TTR) and Supplier HHI data."""
    comp_res = ResilienceAnalyzer.calculate_component_resilience(
        state_container.twin_state['components'],
        state_container.twin_state['shipments']
    )
    hhi = ResilienceAnalyzer.calculate_supplier_hhi(state_container.twin_state['components'])
    return clean_for_json({
        "component_resilience": comp_res,
        "supplier_concentration": hhi
    })


@app.post("/api/simulate/shipping_delay")
def simulate_shipping_delay(req: ShippingDelaySimRequest):
    res = state_container.simulator.simulate_shipping_delay(
        req.component_id, req.additional_delay_days
    )
    res['baseline_trajectory'] = res['baseline_trajectory'].to_dict(orient='records')
    res['simulated_trajectory'] = res['simulated_trajectory'].to_dict(orient='records')
    return clean_for_json(res)


@app.post("/api/simulate/machinery_downtime")
def simulate_machinery_downtime(req: MachineDowntimeSimRequest):
    res = state_container.simulator.simulate_machinery_downtime(
        req.machine_id, req.downtime_days
    )
    return clean_for_json(res)


@app.post("/api/simulate/supplier_disruption")
def simulate_supplier_disruption(req: SupplierDisruptionSimRequest):
    res = state_container.simulator.simulate_supplier_disruption(
        req.component_id, req.supply_reduction_pct, req.extended_delay_days
    )
    res['trajectory'] = res['trajectory'].to_dict(orient='records')
    return clean_for_json(res)


@app.post("/api/simulate/demand_surge")
def simulate_demand_surge(req: DemandSurgeSimRequest):
    res = state_container.simulator.simulate_demand_surge(req.demand_change_pct)
    return clean_for_json(res)


@app.post("/api/simulate/combined")
def simulate_combined(req: CombinedSimRequest):
    res = state_container.simulator.simulate_combined_disruption(
        req.additional_shipping_delay,
        req.machine_downtime_days,
        req.demand_change_pct,
        req.supplier_reduction_pct
    )
    return clean_for_json(res)


@app.get("/api/executive-memo")
def get_executive_memo():
    twin = state_container.twin_state
    roi = PrescriptiveOptimizer.calculate_executive_roi_arbitrage(
        twin['machines'], twin['shipments'], state_container.comp_forecasts
    )
    res = ResilienceAnalyzer.calculate_component_resilience(
        twin['components'], twin['shipments']
    )
    hhi = ResilienceAnalyzer.calculate_supplier_hhi(twin['components'])

    memo = ResilienceAnalyzer.generate_executive_briefing_memo(
        state_container.facility_summary, roi, res, hhi
    )
    recs = CompositeRiskEngine.generate_actionable_recommendations(
        state_container.facility_summary
    )
    return clean_for_json({
        "memo_markdown": memo,
        "recommendations": recs
    })


# Static files mount
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

@app.get("/")
def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "BCG X Digital Twin API is running. Web UI loading..."}

app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
