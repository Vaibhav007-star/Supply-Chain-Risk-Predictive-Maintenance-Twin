"""
Composite Risk Scoring & Actionable Recommendation Engine.
Calculates unified plant-wide health scores and data-driven mitigation advice.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class CompositeRiskEngine:
    """
    Evaluates cross-domain operational risk across equipment, supply chain, and inventory.
    """

    @staticmethod
    def calculate_facility_summary(
        machines_data: List[Dict[str, Any]],
        shipments_data: List[Dict[str, Any]],
        components_forecast: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Produces top-level KPIs and risk distributions for the plant.
        """
        total_machines = len(machines_data)
        critical_machines = [m for m in machines_data if m.get('risk_level') in ['CRITICAL', 'HIGH']]
        
        total_shipments = len(shipments_data)
        at_risk_shipments = [s for s in shipments_data if s.get('risk_tier') in ['CRITICAL', 'HIGH']]
        
        total_components = len(components_forecast)
        stockout_components = [c for c in components_forecast if c.get('stockout_day') is not None]
        safety_breached_components = [c for c in components_forecast if c.get('safety_violation_day') is not None]

        # Financial exposure calculation
        # 1. Potential downtime loss if critical machines fail
        potential_downtime_cost = sum(
            m.get('cost_per_hour_downtime_usd', 650) * 16  # assumed 2 shifts downtime
            for m in critical_machines
        )
        # 2. Inventory deficit replacement / rush freight cost
        potential_rush_cost = sum(
            c.get('cumulative_deficit_units', 0) * 85.0
            for c in stockout_components
        )
        total_exposure = potential_downtime_cost + potential_rush_cost

        # Health Scores (0-100 scale, higher is better)
        m_health = max(0, 100 - (len(critical_machines) / max(1, total_machines) * 100))
        sc_health = max(0, 100 - (len(at_risk_shipments) / max(1, total_shipments) * 100))
        inv_health = max(0, 100 - (len(stockout_components) / max(1, total_components) * 100))

        overall_facility_health = round(0.40 * m_health + 0.35 * sc_health + 0.25 * inv_health, 1)

        return {
            "total_machines": total_machines,
            "critical_machines_count": len(critical_machines),
            "critical_machines": critical_machines,
            "total_shipments": total_shipments,
            "at_risk_shipments_count": len(at_risk_shipments),
            "at_risk_shipments": at_risk_shipments,
            "total_components": total_components,
            "stockout_components_count": len(stockout_components),
            "safety_breached_count": len(safety_breached_components),
            "stockout_components": stockout_components,
            "total_exposure_usd": round(total_exposure, 2),
            "downtime_exposure_usd": round(potential_downtime_cost, 2),
            "logistics_rush_exposure_usd": round(potential_rush_cost, 2),
            "equipment_health_score": round(m_health, 1),
            "supply_chain_health_score": round(sc_health, 1),
            "inventory_health_score": round(inv_health, 1),
            "overall_facility_health": overall_facility_health
        }

    @staticmethod
    def generate_actionable_recommendations(
        facility_summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generates explainable, prioritized recommendations based on detected vulnerabilities.
        """
        recs = []

        # 1. Machine recommendations
        for m in facility_summary.get('critical_machines', []):
            modes = m.get('suspected_failure_modes', ['Degradation risk'])
            recs.append({
                "category": "Maintenance",
                "priority": "HIGH" if m.get('risk_level') == 'CRITICAL' else "MEDIUM",
                "target": f"{m.get('machine_id')} ({m.get('name')})",
                "finding": f"Elevated failure probability ({round(m.get('failure_probability', 0)*100, 1)}%). Diagnosed: {'; '.join(modes)}.",
                "action": f"Schedule immediate preventative inspection during next shift change. Check spindle torque, tool wear ({m.get('tool_wear_min')} min), and lubrication cooling.",
                "estimated_impact": f"Averts ~${m.get('cost_per_hour_downtime_usd', 650)*16:,.0f} in unplanned downtime losses."
            })

        # 2. Logistics & Shipment recommendations
        for s in facility_summary.get('at_risk_shipments', []):
            recs.append({
                "category": "Logistics",
                "priority": "HIGH" if s.get('risk_tier') == 'CRITICAL' else "MEDIUM",
                "target": f"{s.get('shipment_id')} ({s.get('component_name')})",
                "finding": f"Predicted delay: +{s.get('predicted_delay_days', 0)} days on {s.get('origin_region')} route. Composite risk: {round(s.get('composite_delay_risk', 0)*100, 1)}%.",
                "action": f"Contact freight forwarder for {s.get('shipping_mode')} expedited customs clearance or reroute via secondary hub.",
                "estimated_impact": f"Secures replenishment of {s.get('quantity')} units before buffer depletion."
            })

        # 3. Inventory recommendations
        for c in facility_summary.get('stockout_components', []):
            recs.append({
                "category": "Inventory",
                "priority": "CRITICAL",
                "target": f"{c.get('component_id')} ({c.get('component_name')})",
                "finding": f"Stockout projected on Day {c.get('stockout_day')}! Projected deficit: {c.get('cumulative_deficit_units')} units.",
                "action": f"Trigger emergency local spot-buy order and adjust assembly line scheduling to conserve existing stock.",
                "estimated_impact": f"Prevents production halt on downstream manufacturing bays."
            })

        if not recs:
            recs.append({
                "category": "Operations",
                "priority": "LOW",
                "target": "Facility Line 4",
                "finding": "All telemetry, inbound logistics, and safety buffers are currently operating within normal tolerances.",
                "action": "Maintain standard scheduled preventive maintenance and regular lead-time monitoring.",
                "estimated_impact": "Stable baseline operating efficiency."
            })

        return recs
