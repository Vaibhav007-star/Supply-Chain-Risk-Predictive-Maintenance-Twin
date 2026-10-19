"""
Prescriptive Optimization Module for BCG X Supply Chain & Predictive Maintenance Twin.
Provides mathematical decision optimization:
1. Dynamic Safety Stock & EOQ Optimization (Service Level Pareto Optimization)
2. Constrained Maintenance Scheduling Optimizer (Knapsack-style Capacity Optimization)
3. Financial Value-at-Risk (VaR) & Reactive vs Prescriptive ROI Arbitrage
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import norm


class PrescriptiveOptimizer:
    """
    Mathematical decision optimization engine translating predictive outputs
    into cost-optimal operational policies.
    """

    # Standard Z-scores for target service levels (cycle service level)
    SERVICE_LEVEL_Z = {
        0.90: 1.282,
        0.95: 1.645,
        0.98: 2.054,
        0.99: 2.326
    }

    @classmethod
    def optimize_inventory_policies(
        cls,
        components: List[Dict[str, Any]],
        shipments: List[Dict[str, Any]],
        target_service_level: float = 0.95,
        holding_cost_rate: float = 0.20,  # 20% annual holding cost rate
        order_cost_usd: float = 150.0      # administrative fixed cost per order
    ) -> Dict[str, Any]:
        """
        Calculates mathematically optimal Safety Stock, Reorder Point (ROP),
        and Economic Order Quantity (EOQ) using stochastic demand and lead-time variability.
        """
        z_score = cls.SERVICE_LEVEL_Z.get(target_service_level, 1.645)
        results = []
        total_current_holding_cost = 0.0
        total_optimized_holding_cost = 0.0
        total_annual_order_cost = 0.0
        working_capital_freed = 0.0

        for comp in components:
            c_id = comp['component_id']
            name = comp['name']
            d_mean = comp.get('daily_burn_rate', 25)
            # Stochastic variance assumptions based on industrial manufacturing lines
            d_std = max(1.0, d_mean * 0.18)  # 18% coefficient of variation for daily demand
            
            # Find matching shipment for lead time
            shp = next((s for s in shipments if s.get('component_id') == c_id), None)
            l_mean = shp.get('scheduled_days', 6) if shp else comp.get('scheduled_lead_days', 6)
            l_delay = shp.get('predicted_delay_days', 0) if shp else 0
            effective_lead_time = l_mean + (l_delay * 0.5)  # partially expected lead time
            l_std = max(0.5, shp.get('predicted_delay_days', 1.0) * 0.6) if shp else 1.0

            unit_cost = comp.get('unit_cost_usd', 100.0)
            annual_demand = d_mean * 365
            annual_unit_holding_cost = unit_cost * holding_cost_rate

            # 1. Economic Order Quantity (Wilson EOQ Formula): sqrt((2 * D * S) / H)
            if annual_unit_holding_cost > 0:
                optimal_eoq = int(np.round(np.sqrt((2 * annual_demand * order_cost_usd) / annual_unit_holding_cost)))
            else:
                optimal_eoq = comp.get('active_order_qty', 200)

            # 2. Optimal Safety Stock formula under demand and lead-time stochasticity:
            # SS = Z * sqrt( L * sigma_D^2 + D^2 * sigma_L^2 )
            variance_term = (effective_lead_time * (d_std ** 2)) + ((d_mean ** 2) * (l_std ** 2))
            optimal_safety_stock = int(np.round(z_score * np.sqrt(variance_term)))

            # 3. Optimal Reorder Point (ROP) = (D * L) + SS
            optimal_rop = int(np.round((d_mean * effective_lead_time) + optimal_safety_stock))

            current_ss = comp.get('safety_stock', optimal_safety_stock)
            current_rop = comp.get('reorder_point', optimal_rop)
            current_order_qty = comp.get('active_order_qty', optimal_eoq)

            # Annualized Financial Trade-offs
            curr_holding = current_ss * annual_unit_holding_cost
            opt_holding = optimal_safety_stock * annual_unit_holding_cost
            annual_orders = annual_demand / max(1, optimal_eoq)
            annual_order_cost = annual_orders * order_cost_usd

            wc_delta = (current_ss - optimal_safety_stock) * unit_cost
            working_capital_freed += wc_delta
            total_current_holding_cost += curr_holding
            total_optimized_holding_cost += opt_holding
            total_annual_order_cost += annual_order_cost

            results.append({
                "component_id": c_id,
                "name": name,
                "unit_cost_usd": unit_cost,
                "daily_demand": d_mean,
                "lead_time_days": round(effective_lead_time, 1),
                "current_safety_stock": current_ss,
                "optimized_safety_stock": optimal_safety_stock,
                "safety_stock_delta": optimal_safety_stock - current_ss,
                "current_reorder_point": current_rop,
                "optimized_reorder_point": optimal_rop,
                "current_order_qty": current_order_qty,
                "optimized_eoq": optimal_eoq,
                "annual_holding_cost_optimized_usd": round(opt_holding, 2),
                "working_capital_impact_usd": round(wc_delta, 2),
                "status": "Over-buffered (Excess Capital)" if optimal_safety_stock < current_ss else (
                    "Under-buffered (Stockout Risk)" if optimal_safety_stock > current_ss else "Optimal"
                )
            })

        return {
            "target_service_level": target_service_level,
            "components": results,
            "total_current_holding_cost_usd": round(total_current_holding_cost, 2),
            "total_optimized_holding_cost_usd": round(total_optimized_holding_cost, 2),
            "net_annual_holding_savings_usd": round(total_current_holding_cost - total_optimized_holding_cost, 2),
            "net_working_capital_freed_usd": round(working_capital_freed, 2),
            "total_annual_order_cost_usd": round(total_annual_order_cost, 2)
        }

    @classmethod
    def optimize_maintenance_schedule(
        cls,
        machines: List[Dict[str, Any]],
        available_technician_hours: int = 24,
        cost_per_tech_hour_usd: float = 85.0
    ) -> Dict[str, Any]:
        """
        Constrained multi-asset maintenance scheduling optimizer.
        Solves the resource-constrained maintenance problem to maximize Avoided Downtime Loss ($)
        within available technician capacity.
        """
        candidates = []
        for m in machines:
            m_id = m['machine_id']
            prob = m.get('failure_probability', 0.05)
            hourly_downtime_cost = m.get('cost_per_hour_downtime_usd', 650.0)
            
            # Planned maintenance duration is typically 3-5 hours, whereas unplanned breakdown causes 12-24 hours
            unplanned_downtime_hours = 16.0
            planned_maintenance_duration_hours = 4.0
            
            expected_unplanned_loss = prob * (hourly_downtime_loss := hourly_downtime_cost * unplanned_downtime_hours)
            planned_cost = (planned_maintenance_duration_hours * cost_per_tech_hour_usd) + 250.0  # consumables/parts
            
            # Net value of proactive intervention = Expected Unplanned Loss Avoided - Planned Intervention Cost
            net_benefit = expected_unplanned_loss - planned_cost
            efficiency_ratio = net_benefit / planned_maintenance_duration_hours  # ROI per technician hour

            candidates.append({
                "machine_id": m_id,
                "name": m.get('name', 'Asset'),
                "bay": m.get('bay', 'Line 4'),
                "failure_probability": prob,
                "risk_level": m.get('risk_level', 'HEALTHY'),
                "required_tech_hours": planned_maintenance_duration_hours,
                "planned_cost_usd": planned_cost,
                "expected_unplanned_loss_usd": expected_unplanned_loss,
                "net_avoided_loss_usd": net_benefit,
                "efficiency_score": efficiency_ratio,
                "urgency_rank": 1 if m.get('risk_level') == 'CRITICAL' else (2 if m.get('risk_level') == 'HIGH' else 3)
            })

        # Sort primarily by urgency rank and secondarily by net efficiency ratio
        candidates = sorted(candidates, key=lambda x: (x['urgency_rank'], -x['efficiency_score']))

        scheduled = []
        deferred = []
        used_hours = 0.0
        total_avoided_losses = 0.0
        total_intervention_investment = 0.0

        for cand in candidates:
            if cand['failure_probability'] >= 0.15 and (used_hours + cand['required_tech_hours'] <= available_technician_hours):
                scheduled.append(cand)
                used_hours += cand['required_tech_hours']
                total_avoided_losses += cand['expected_unplanned_loss_usd']
                total_intervention_investment += cand['planned_cost_usd']
            else:
                deferred.append(cand)

        net_savings = total_avoided_losses - total_intervention_investment
        roi_percent = (net_savings / total_intervention_investment * 100) if total_intervention_investment > 0 else 0.0

        return {
            "available_technician_hours": available_technician_hours,
            "allocated_technician_hours": used_hours,
            "scheduled_interventions_count": len(scheduled),
            "scheduled_interventions": scheduled,
            "deferred_assets": deferred,
            "total_unplanned_losses_avoided_usd": round(total_avoided_losses, 2),
            "total_intervention_investment_usd": round(total_intervention_investment, 2),
            "net_maintenance_savings_usd": round(net_savings, 2),
            "roi_percentage": round(roi_percent, 1)
        }

    @classmethod
    def calculate_executive_roi_arbitrage(
        cls,
        machines: List[Dict[str, Any]],
        shipments: List[Dict[str, Any]],
        components_forecast: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        BCG X C-Suite Value Quantification:
        Compares 'Reactive Status Quo' vs 'Antigravity Prescriptive AI Operations'.
        Computes Total Value-at-Risk (VaR), Avoided Losses, Net ROI, and Payback Horizon.
        """
        # 1. Reactive Status Quo Losses:
        # - Catastrophic breakdown costs with emergency rush parts ($1,200/hr + 24hr downtime)
        # - Stockout production halts & expedited air freight ($220/unit penalty)
        critical_m = [m for m in machines if m.get('risk_level') in ['CRITICAL', 'HIGH']]
        stockout_c = [c for c in components_forecast if c.get('stockout_day') is not None]

        reactive_downtime_loss = sum(
            m.get('cost_per_hour_downtime_usd', 650) * 20.0 * max(0.4, m.get('failure_probability', 0.5))
            for m in critical_m
        )
        reactive_stockout_loss = sum(
            c.get('cumulative_deficit_units', 50) * 220.0  # severe late order cancellation penalty
            for c in stockout_c
        )
        reactive_rush_logistics = sum(
            s.get('quantity', 100) * 18.5  # emergency courier expediting
            for s in shipments if s.get('risk_tier') in ['CRITICAL', 'HIGH']
        )
        total_reactive_cost = reactive_downtime_loss + reactive_stockout_loss + reactive_rush_logistics

        # 2. Prescriptive Operations (With Antigravity Twin):
        # - Planned maintenance during non-peak hours (4 hrs @ $85/hr + $250 parts)
        # - Dynamic safety stock rebalancing (holding cost of modest buffer)
        # - Scheduled dual-sourcing / standard expediting
        prescriptive_m_cost = len(critical_m) * (4.0 * 85.0 + 250.0)
        prescriptive_buffer_cost = len(stockout_c) * 120.0  # carrying cost of safety buffer
        prescriptive_logistics_cost = len([s for s in shipments if s.get('risk_tier') in ['CRITICAL', 'HIGH']]) * 350.0  # proactive reroute fee
        total_prescriptive_cost = prescriptive_m_cost + prescriptive_buffer_cost + prescriptive_logistics_cost

        net_savings = max(0.0, total_reactive_cost - total_prescriptive_cost)
        roi_multiplier = (total_reactive_cost / max(1.0, total_prescriptive_cost))
        annualized_net_savings = net_savings * 4.0  # scaled quarterly projection to annual EBITDA

        return {
            "total_value_at_risk_usd": round(total_reactive_cost, 2),
            "reactive_downtime_loss_usd": round(reactive_downtime_loss, 2),
            "reactive_stockout_loss_usd": round(reactive_stockout_loss, 2),
            "reactive_rush_logistics_usd": round(reactive_rush_logistics, 2),
            "total_prescriptive_cost_usd": round(total_prescriptive_cost, 2),
            "prescriptive_maintenance_cost_usd": round(prescriptive_m_cost, 2),
            "prescriptive_buffer_cost_usd": round(prescriptive_buffer_cost, 2),
            "net_quarterly_savings_usd": round(net_savings, 2),
            "annualized_ebitda_impact_usd": round(annualized_net_savings, 2),
            "roi_multiplier": round(roi_multiplier, 1),
            "payback_period_weeks": round(max(0.8, 52.0 / (roi_multiplier * 2.5)), 1)
        }
