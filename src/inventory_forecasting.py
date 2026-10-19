"""
Inventory Forecasting & Stockout Projection Module.
Calculates deterministic forward inventory trajectories, stockout dates, and safety stock violations.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class InventoryForecaster:
    """
    Simulates day-by-day inventory level changes over a planning horizon (default 30 days).
    """

    @staticmethod
    def forecast_component_trajectory(
        component: Dict[str, Any],
        inbound_shipment: Dict[str, Any] = None,
        horizon_days: int = 30,
        demand_multiplier: float = 1.0,
        shipping_delay_days: int = 0,
        supply_reduction_pct: float = 0.0,
        safety_stock_override: int = None
    ) -> Dict[str, Any]:
        """
        Calculates daily inventory balances, stockout events, and deficit volumes.
        """
        current_stock = component['current_stock']
        daily_burn = component['daily_burn_rate'] * demand_multiplier
        safety_stock = safety_stock_override if safety_stock_override is not None else component['safety_stock']
        reorder_point = component['reorder_point']

        # Determine inbound delivery arrival day
        arrival_day = None
        inbound_qty = 0
        if inbound_shipment:
            base_lead = inbound_shipment.get('scheduled_days', component.get('scheduled_lead_days', 7))
            arrival_day = base_lead + shipping_delay_days
            inbound_qty = inbound_shipment.get('quantity', component.get('active_order_qty', 100))
            inbound_qty = int(inbound_qty * (1.0 - supply_reduction_pct / 100.0))

        days = []
        stock_levels = []
        burn_rates = []
        inflow_events = []
        stockout_date_day = None
        safety_violation_day = None
        min_stock_reached = current_stock
        stockout_days_count = 0
        cumulative_deficit = 0

        running_stock = current_stock

        for day in range(1, horizon_days + 1):
            daily_inflow = 0
            if arrival_day is not None and day == arrival_day:
                daily_inflow = inbound_qty

            running_stock += daily_inflow
            running_stock -= daily_burn

            if running_stock < safety_stock and safety_violation_day is None:
                safety_violation_day = day

            if running_stock <= 0:
                stockout_days_count += 1
                cumulative_deficit += abs(running_stock)
                if stockout_date_day is None:
                    stockout_date_day = day
                running_stock = 0  # physical stock cannot be negative

            if running_stock < min_stock_reached:
                min_stock_reached = running_stock

            days.append(f"Day {day}")
            stock_levels.append(round(running_stock, 1))
            burn_rates.append(round(daily_burn, 1))
            inflow_events.append(daily_inflow)

        trajectory_df = pd.DataFrame({
            "day": days,
            "day_num": list(range(1, horizon_days + 1)),
            "inventory_level": stock_levels,
            "burn_rate": burn_rates,
            "inflow": inflow_events,
            "safety_stock_threshold": [safety_stock] * horizon_days,
            "reorder_point": [reorder_point] * horizon_days
        })

        # Calculate run-out days if no inbound arrives
        days_of_supply = round(current_stock / (daily_burn + 1e-4), 1)

        return {
            "component_id": component['component_id'],
            "component_name": component['name'],
            "current_stock": current_stock,
            "days_of_supply": days_of_supply,
            "safety_stock": safety_stock,
            "stockout_day": stockout_date_day,
            "safety_violation_day": safety_violation_day,
            "stockout_days_count": stockout_days_count,
            "cumulative_deficit_units": round(cumulative_deficit, 1),
            "min_stock_reached": round(min_stock_reached, 1),
            "inbound_arrival_day": arrival_day,
            "inbound_delivered_qty": inbound_qty,
            "trajectory": trajectory_df
        }
