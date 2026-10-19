"""
Integration and End-to-End API Test Suite for BCG X Digital Twin FastAPI Server.
Validates all REST endpoints, ML model integration, simulations, and static web serving.
"""
import pytest
from starlette.testclient import TestClient
from server import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"]["predictive_maintenance"] is True
    assert data["models_loaded"]["supply_chain_delay"] is True
    assert data["total_assets"] > 0


def test_api_state():
    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert "facility_summary" in data
    assert "roi_arbitrage" in data
    assert "resilience_overview" in data
    assert data["facility_summary"]["overall_facility_health"] > 0
    assert data["roi_arbitrage"]["roi_multiplier"] > 0


def test_api_machines_and_detail():
    response = client.get("/api/machines")
    assert response.status_code == 200
    machines = response.json()
    assert len(machines) > 0
    first_id = machines[0]["machine_id"]

    # Detail with XAI
    detail_res = client.get(f"/api/machine/{first_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert "machine" in detail
    assert "explainability" in detail
    assert "contributions" in detail["explainability"]


def test_api_shipments_and_detail():
    response = client.get("/api/shipments")
    assert response.status_code == 200
    shipments = response.json()
    assert len(shipments) > 0
    first_id = shipments[0]["shipment_id"]

    # Detail with XAI
    detail_res = client.get(f"/api/shipment/{first_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert "shipment" in detail
    assert "explainability" in detail
    assert "contributions" in detail["explainability"]


def test_api_inventory():
    response = client.get("/api/inventory")
    assert response.status_code == 200
    components = response.json()
    assert len(components) > 0
    assert "trajectory" in components[0]
    assert len(components[0]["trajectory"]) == 30


def test_api_optimize_inventory():
    payload = {
        "target_service_level": 0.98,
        "order_cost_usd": 150.0
    }
    response = client.post("/api/optimize/inventory", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_service_level"] == 0.98
    assert len(data["components"]) > 0
    assert data["total_optimized_holding_cost_usd"] > 0


def test_api_optimize_maintenance():
    payload = {
        "available_technician_hours": 32,
        "cost_per_tech_hour_usd": 85.0
    }
    response = client.post("/api/optimize/maintenance", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["available_technician_hours"] == 32
    assert data["total_unplanned_losses_avoided_usd"] > 0


def test_api_resilience():
    response = client.get("/api/resilience")
    assert response.status_code == 200
    data = response.json()
    assert "component_resilience" in data
    assert "supplier_concentration" in data
    assert len(data["component_resilience"]["resilience_records"]) > 0
    assert data["supplier_concentration"]["hhi_index"] > 0


def test_api_simulations():
    # 1. Shipping Delay
    r1 = client.post("/api/simulate/shipping_delay", json={"component_id": "CMP-101", "additional_delay_days": 8})
    assert r1.status_code == 200
    assert "simulated_arrival_day" in r1.json()

    # 2. Machinery Downtime
    r2 = client.post("/api/simulate/machinery_downtime", json={"machine_id": "CNC-101", "downtime_days": 4})
    assert r2.status_code == 200
    assert r2.json()["lost_production_units"] > 0

    # 3. Supplier Disruption
    r3 = client.post("/api/simulate/supplier_disruption", json={"component_id": "CMP-101", "supply_reduction_pct": 50, "extended_delay_days": 5})
    assert r3.status_code == 200
    assert "trajectory" in r3.json()

    # 4. Demand Surge
    r4 = client.post("/api/simulate/demand_surge", json={"demand_change_pct": 35})
    assert r4.status_code == 200
    assert "components_with_stockout" in r4.json()

    # 5. Combined Disruption
    r5 = client.post("/api/simulate/combined", json={
        "additional_shipping_delay": 5,
        "machine_downtime_days": 3,
        "demand_change_pct": 20,
        "supplier_reduction_pct": 25
    })
    assert r5.status_code == 200
    assert "comparison" in r5.json()


def test_api_executive_memo():
    response = client.get("/api/executive-memo")
    assert response.status_code == 200
    data = response.json()
    assert "memo_markdown" in data
    assert "BCG X EXECUTIVE BRIEFING" in data["memo_markdown"]
    assert len(data["recommendations"]) > 0


def test_serve_index_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "BCG X" in response.text
