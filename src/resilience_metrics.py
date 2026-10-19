"""
Supply Chain Resilience & Network Vulnerability Framework (BCG Standard).
Implements:
1. Time-to-Survive (TTS) vs Time-to-Recover (TTR) Analysis
2. Resilience Gap & Critical Node Vulnerability Index
3. Sourcing Concentration Herfindahl-Hirschman Index (HHI)
4. Automated Boardroom Executive Memo Generator
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class ResilienceAnalyzer:
    """
    Evaluates facility resilience against supply disruptions and equipment failure
    using the BCG Time-to-Survive (TTS) / Time-to-Recover (TTR) methodology.
    """

    @classmethod
    def calculate_component_resilience(
        cls,
        components: List[Dict[str, Any]],
        shipments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculates TTS, TTR, and Resilience Gap for every inbound component.
        """
        records = []
        critical_count = 0

        for comp in components:
            c_id = comp['component_id']
            name = comp['name']
            stock = comp.get('current_stock', 100)
            burn = max(1, comp.get('daily_burn_rate', 10))
            supplier = comp.get('supplier', 'Unknown')
            route = comp.get('primary_route', 'Global')

            # Time to Survive (Days before complete component starvation)
            tts_days = round(stock / burn, 1)

            # Find matching shipment to estimate Time to Recover (Days to emergency replace/reroute)
            shp = next((s for s in shipments if s.get('component_id') == c_id), None)
            base_lead = shp.get('scheduled_days', 6) if shp else comp.get('scheduled_lead_days', 6)
            pred_delay = shp.get('predicted_delay_days', 0) if shp else 0
            
            # TTR includes base lead time + spot expediting qualification buffer (typically +4 to +6 days)
            ttr_days = round(base_lead + pred_delay + 4.0, 1)

            resilience_gap = round(tts_days - ttr_days, 1)

            if resilience_gap < 0:
                classification = "VULNERABLE (TTS < TTR)"
                quadrant = "Critical Shortfall"
                resilience_score = max(10.0, 50.0 + (resilience_gap * 5))
                critical_count += 1
            elif resilience_gap <= 3:
                classification = "TIGHT BUFFER (Near-zero margin)"
                quadrant = "Buffer Warning"
                resilience_score = 70.0
            else:
                classification = "RESILIENT (TTS >= TTR)"
                quadrant = "Resilient Domain"
                resilience_score = min(100.0, 75.0 + (resilience_gap * 2))

            records.append({
                "component_id": c_id,
                "name": name,
                "supplier": supplier,
                "route": route,
                "current_stock": stock,
                "daily_burn": burn,
                "time_to_survive_days": tts_days,
                "time_to_recover_days": ttr_days,
                "resilience_gap_days": resilience_gap,
                "classification": classification,
                "quadrant": quadrant,
                "resilience_score": round(resilience_score, 1)
            })

        avg_resilience_score = round(float(np.mean([r['resilience_score'] for r in records])), 1)

        return {
            "resilience_records": records,
            "vulnerable_components_count": critical_count,
            "total_components": len(records),
            "average_network_resilience_score": avg_resilience_score
        }

    @classmethod
    def calculate_supplier_hhi(
        cls,
        components: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Computes the Herfindahl-Hirschman Index (HHI) of supplier concentration
        based on total annualized procurement value.
        """
        supplier_spend = {}
        total_spend = 0.0

        for comp in components:
            annual_units = comp.get('daily_burn_rate', 20) * 365
            spend = annual_units * comp.get('unit_cost_usd', 100.0)
            supplier = comp.get('supplier', 'Unknown').split(" (")[0]
            supplier_spend[supplier] = supplier_spend.get(supplier, 0.0) + spend
            total_spend += spend

        shares = []
        hhi = 0.0
        for sup, spend in supplier_spend.items():
            market_share_pct = (spend / max(1.0, total_spend)) * 100.0
            hhi += market_share_pct ** 2
            shares.append({
                "supplier": sup,
                "annual_spend_usd": round(spend, 2),
                "spend_share_pct": round(market_share_pct, 1)
            })

        shares = sorted(shares, key=lambda x: x['spend_share_pct'], reverse=True)

        if hhi < 1500:
            concentration_tier = "DIVERSIFIED (Low Structural Risk)"
        elif hhi <= 2500:
            concentration_tier = "MODERATE CONCENTRATION"
        else:
            concentration_tier = "HIGH CONCENTRATION (Single-Point Sourcing Exposure)"

        return {
            "hhi_index": round(hhi, 1),
            "concentration_tier": concentration_tier,
            "total_annual_spend_usd": round(total_spend, 2),
            "supplier_breakdown": shares
        }

    @classmethod
    def generate_executive_briefing_memo(
        cls,
        facility_summary: Dict[str, Any],
        roi_arbitrage: Dict[str, Any],
        resilience_data: Dict[str, Any],
        hhi_data: Dict[str, Any]
    ) -> str:
        """
        Produces a C-Suite executive briefing memo in BCG consulting standard.
        """
        memo = f"""# 📑 BCG X EXECUTIVE BRIEFING: SUPPLY CHAIN & DIGITAL TWIN RESILIENCE
**Target Entity**: {facility_summary.get('facility_name', 'Advanced Manufacturing Facility - Line 4')}  
**Evaluation Horizon**: 30-Day Forward Digital Twin Horizon  
**Strategic Classification**: Board-Level Operational Risk & EBITDA Optimization Memo  

---

### 1. EXECUTIVE SUMMARY & VALUE CAPTURE
An integrated AI digital twin assessment across physical machine telemetry, tier-1 supplier lead times, and multi-echelon inventory trajectories has detected **${roi_arbitrage.get('total_value_at_risk_usd', 0):,.0f} in near-term Value-at-Risk (VaR)**.

By transitioning from reactive maintenance and emergency spot-freight buying to **Antigravity Prescriptive AI Operations**, the facility captures:
- **Net Annualized EBITDA Savings**: **${roi_arbitrage.get('annualized_ebitda_impact_usd', 0):,.0f} USD**
- **Prescriptive ROI Multiplier**: **{roi_arbitrage.get('roi_multiplier', 0):.1f}x Payback Ratio**
- **Capital Payback Horizon**: **{roi_arbitrage.get('payback_period_weeks', 0):.1f} Weeks**
- **Plant Composite Health Score**: **{facility_summary.get('overall_facility_health', 0)}%**

---

### 2. RESILIENCE PROFILE (TIME-TO-SURVIVE vs TIME-TO-RECOVER)
- **Facility Average Resilience Index**: **{resilience_data.get('average_network_resilience_score', 0)} / 100**
- **Critical Vulnerability Exposure**: **{resilience_data.get('vulnerable_components_count', 0)} of {resilience_data.get('total_components', 0)} components** exhibit a negative resilience gap ($\text{{TTS}} < \text{{TTR}}$), indicating that an unexpected upstream supply shock will force line shutdowns before replacement stock can be secured.
- **Supplier Concentration (HHI)**: **{hhi_data.get('hhi_index', 0):,.0f}** ({hhi_data.get('concentration_tier', 'Moderate')}) across **${hhi_data.get('total_annual_spend_usd', 0):,.0f}** total annual spend.

---

### 3. OPERATIONAL HOTSPOTS & IMMEDIATE MITIGATIONS (DAYS 0 - 7)
"""
        # Critical machines
        if facility_summary.get('critical_machines_count', 0) > 0:
            memo += "#### 🚨 Critical Machinery Interventions:\n"
            for m in facility_summary.get('critical_machines', [])[:2]:
                memo += f"- **{m.get('machine_id')} ({m.get('name')})**: Failure probability `{round(m.get('failure_probability', 0)*100, 1)}%`. Tool wear at `{m.get('tool_wear_min')} min`. Immediate scheduled downtime during shift change saves **${m.get('cost_per_hour_downtime_usd', 650)*16:,.0f}** in catastrophic downtime.\n"

        # Stockout components
        if facility_summary.get('stockout_components_count', 0) > 0:
            memo += "\n#### 📦 Imminent Stockout Vulnerabilities:\n"
            for c in facility_summary.get('stockout_components', [])[:2]:
                memo += f"- **{c.get('component_name')}**: Projected stockout on **Day {c.get('stockout_day')}** with a cumulative deficit of **{c.get('cumulative_deficit_units')} units**. Action: Trigger expedited local air spot-freight before Day {max(1, c.get('stockout_day', 5)-2)}.\n"

        memo += """
---

### 4. STRATEGIC TRANSFORMATION ROADMAP (30 - 90 DAYS)
1. **Dynamic Safety Stock Rebalancing**: Shift static min-max inventory thresholds to stochastic lead-time buffers, unlocking working capital while shielding against climate/weather transit variance.
2. **Dual-Sourcing Qualification**: Onboard nearshore secondary suppliers for nodes where $\\text{TTS} < \\text{TTR}$, reducing replenishment lead times by up to 45%.
3. **Condition-Based Maintenance Protocol**: Embed real-time telemetry threshold triggers directly into ERP work-order dispatch systems to eliminate unplanned spindle burnouts.

*Prepared for BCG X Tech & Digital Operations Leadership Review.*
"""
        return memo
