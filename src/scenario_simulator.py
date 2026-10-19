"""
Digital Twin What-If Scenario Simulator.
Computes deterministic operational outcomes for shipping delays, machine downtime,
supplier disruptions, demand surges, and combined multi-vector shocks.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.inventory_forecasting import InventoryForecaster


class ScenarioSimulator:
    """
    Simulates operational impacts across factory production lines and supply chains.
    """

    def __init__(self, twin_state: Dict[str, Any]):
        self.twin_state = twin_state
        self.machines = twin_state.get('machines', [])
        self.components = twin_state.get('components', [])
        self.shipments = twin_state.get('shipments', [])

    def simulate_shipping_delay(
        self,
        target_component_id: str,
        additional_delay_days: int = 5
    ) -> Dict[str, Any]:
        """
        Scenario 1: Shipping Delay (0 to 30 days).
        Calculates revised arrival, stockout date, deficit, and production impact.
        """
        comp = next((c for c in self.components if c['component_id'] == target_component_id), self.components[0])
        shipment = next((s for s in self.shipments if s['component_id'] == comp['component_id']), None)

        baseline_forecast = InventoryForecaster.forecast_component_trajectory(
            comp, shipment, shipping_delay_days=0
        )
        simulated_forecast = InventoryForecaster.forecast_component_trajectory(
            comp, shipment, shipping_delay_days=additional_delay_days
        )

        # Production impact
        daily_burn = comp['daily_burn_rate']
        deficit = simulated_forecast['cumulative_deficit_units']
        lost_production_batches = round(deficit / max(1, daily_burn), 1)

        mitigations = []
        if simulated_forecast['stockout_day'] is not None:
            mitigations.append(f"Stockout occurs on Day {simulated_forecast['stockout_day']}. Expedite shipment via Air Freight.")
            mitigations.append("Borrow buffer stock from secondary plant warehouse if available.")
            mitigations.append("Throttle daily production down by 15% to stretch existing inventory.")
        else:
            mitigations.append("Current inventory + safety stock buffer absorbs this delay without line stoppage.")

        return {
            "scenario": "Shipping Delay",
            "component_id": comp['component_id'],
            "component_name": comp['name'],
            "additional_delay_days": additional_delay_days,
            "baseline_arrival_day": baseline_forecast['inbound_arrival_day'],
            "simulated_arrival_day": simulated_forecast['inbound_arrival_day'],
            "baseline_stockout_day": baseline_forecast['stockout_day'],
            "simulated_stockout_day": simulated_forecast['stockout_day'],
            "inventory_deficit_units": deficit,
            "estimated_production_days_lost": lost_production_batches,
            "mitigation_actions": mitigations,
            "baseline_trajectory": baseline_forecast['trajectory'],
            "simulated_trajectory": simulated_forecast['trajectory']
        }

    def simulate_machinery_downtime(
        self,
        target_machine_id: str,
        downtime_days: int = 3,
        daily_hours_down: int = 16
    ) -> Dict[str, Any]:
        """
        Scenario 2: Machinery Failure & Downtime.
        Calculates lost capacity, production backlog, component consumption freeze, and financial losses.
        """
        machine = next((m for m in self.machines if m['machine_id'] == target_machine_id), self.machines[0])
        total_downtime_hours = downtime_days * daily_hours_down

        cost_per_hour = machine.get('cost_per_hour_downtime_usd', 650)
        daily_cap = machine.get('daily_capacity_units', 250)

        # Capacity lost
        units_lost = int(daily_cap * downtime_days)
        financial_loss = total_downtime_hours * cost_per_hour

        # Related component burn rate changes
        related_comp_name = machine.get('component', 'General Consumable')
        comp = next((c for c in self.components if c['name'] == related_comp_name), None)
        comp_burn_saved = (comp['daily_burn_rate'] * downtime_days) if comp else 0

        recovery_options = [
            f"Activate idle capacity in {machine.get('bay')} to absorb {int(units_lost * 0.6)} backlog units.",
            f"Schedule emergency technical repair squad for {machine.get('name')}.",
            "Shift unmachined billet workpieces to partner subcontractor facility."
        ]

        return {
            "scenario": "Machinery Failure",
            "machine_id": machine['machine_id'],
            "machine_name": machine['name'],
            "downtime_days": downtime_days,
            "total_downtime_hours": total_downtime_hours,
            "lost_production_units": units_lost,
            "backlog_units_accumulated": units_lost,
            "estimated_downtime_cost_usd": financial_loss,
            "component_consumption_halted_units": comp_burn_saved,
            "recovery_options": recovery_options
        }

    def simulate_supplier_disruption(
        self,
        target_component_id: str,
        supply_reduction_pct: float = 50.0,
        extended_delay_days: int = 7
    ) -> Dict[str, Any]:
        """
        Scenario 3: Supplier Disruption (Reduced quantity or prolonged supply shock).
        """
        comp = next((c for c in self.components if c['component_id'] == target_component_id), self.components[0])
        shipment = next((s for s in self.shipments if s['component_id'] == comp['component_id']), None)

        sim_forecast = InventoryForecaster.forecast_component_trajectory(
            comp, shipment,
            shipping_delay_days=extended_delay_days,
            supply_reduction_pct=supply_reduction_pct
        )

        return {
            "scenario": "Supplier Disruption",
            "component_id": comp['component_id'],
            "component_name": comp['name'],
            "supplier": comp['supplier'],
            "supply_reduction_pct": supply_reduction_pct,
            "extended_delay_days": extended_delay_days,
            "stockout_day": sim_forecast['stockout_day'],
            "cumulative_deficit_units": sim_forecast['cumulative_deficit_units'],
            "min_stock_reached": sim_forecast['min_stock_reached'],
            "recommended_strategy": [
                f"Qualify pre-vetted secondary supplier for {comp['name']}.",
                f"Increase safety stock buffer from {comp['safety_stock']} to {int(comp['safety_stock'] * 1.5)} units.",
                f"Place dual-source split orders (70/30 allocation) to prevent single-supplier dependency."
            ],
            "trajectory": sim_forecast['trajectory']
        }

    def simulate_demand_surge(
        self,
        demand_change_pct: float = 30.0
    ) -> Dict[str, Any]:
        """
        Scenario 4: Unexpected Demand Surge (-50% to +100%).
        """
        multiplier = 1.0 + (demand_change_pct / 100.0)
        component_impacts = []

        for comp in self.components:
            shipment = next((s for s in self.shipments if s['component_id'] == comp['component_id']), None)
            forecast = InventoryForecaster.forecast_component_trajectory(
                comp, shipment, demand_multiplier=multiplier
            )
            base_forecast = InventoryForecaster.forecast_component_trajectory(
                comp, shipment, demand_multiplier=1.0
            )

            component_impacts.append({
                "component_id": comp['component_id'],
                "name": comp['name'],
                "baseline_burn": comp['daily_burn_rate'],
                "new_burn": round(comp['daily_burn_rate'] * multiplier, 1),
                "baseline_stockout": base_forecast['stockout_day'],
                "simulated_stockout": forecast['stockout_day'],
                "deficit_units": forecast['cumulative_deficit_units'],
                "safety_stock_recommendation": int(comp['safety_stock'] * multiplier)
            })

        stockout_count = sum(1 for c in component_impacts if c['simulated_stockout'] is not None)

        return {
            "scenario": "Demand Surge",
            "demand_change_pct": demand_change_pct,
            "demand_multiplier": multiplier,
            "components_with_stockout": stockout_count,
            "component_details": component_impacts
        }

    def simulate_combined_disruption(
        self,
        additional_shipping_delay: int = 6,
        machine_downtime_days: int = 4,
        demand_change_pct: float = 25.0,
        supplier_reduction_pct: float = 30.0
    ) -> Dict[str, Any]:
        """
        Scenario 5: Combined Multi-Vector Disruption.
        Compares baseline operations against a concurrent multi-shock stress test.
        """
        demand_mult = 1.0 + (demand_change_pct / 100.0)

        # Baseline metrics
        baseline_deficits = 0
        baseline_stockout_comps = 0
        for comp in self.components:
            shipment = next((s for s in self.shipments if s['component_id'] == comp['component_id']), None)
            bf = InventoryForecaster.forecast_component_trajectory(comp, shipment)
            baseline_deficits += bf['cumulative_deficit_units']
            if bf['stockout_day'] is not None:
                baseline_stockout_comps += 1

        # Simulated combined multi-shock
        sim_deficits = 0
        sim_stockout_comps = 0
        combined_details = []

        for comp in self.components:
            shipment = next((s for s in self.shipments if s['component_id'] == comp['component_id']), None)
            sf = InventoryForecaster.forecast_component_trajectory(
                comp, shipment,
                demand_multiplier=demand_mult,
                shipping_delay_days=additional_shipping_delay,
                supply_reduction_pct=supplier_reduction_pct
            )
            sim_deficits += sf['cumulative_deficit_units']
            if sf['stockout_day'] is not None:
                sim_stockout_comps += 1
            combined_details.append({
                "component_id": comp['component_id'],
                "component_name": comp['name'],
                "simulated_stockout_day": sf['stockout_day'],
                "deficit_units": sf['cumulative_deficit_units']
            })

        # Downtime losses
        downtime_cost = sum(
            m.get('cost_per_hour_downtime_usd', 650) * 16 * machine_downtime_days
            for m in self.machines[:3]  # impact on first 3 critical lines
        )
        lost_production_units = sum(
            m.get('daily_capacity_units', 250) * machine_downtime_days
            for m in self.machines[:3]
        )

        total_financial_exposure = downtime_cost + (sim_deficits * 85.0)

        return {
            "scenario": "Combined Disruption",
            "parameters": {
                "shipping_delay_days": additional_shipping_delay,
                "machine_downtime_days": machine_downtime_days,
                "demand_change_pct": demand_change_pct,
                "supplier_reduction_pct": supplier_reduction_pct
            },
            "comparison": {
                "baseline_stockout_components": baseline_stockout_comps,
                "simulated_stockout_components": sim_stockout_comps,
                "baseline_inventory_deficit": round(baseline_deficits, 1),
                "simulated_inventory_deficit": round(sim_deficits, 1),
                "lost_production_units": lost_production_units,
                "lost_production_cost_usd": round(downtime_cost, 2),
                "total_combined_financial_exposure_usd": round(total_financial_exposure, 2)
            },
            "component_breakdown": combined_details
        }
