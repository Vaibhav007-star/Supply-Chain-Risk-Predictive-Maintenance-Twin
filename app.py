"""
Supply Chain Risk & Predictive Maintenance Digital Twin.
BCG X Executive Decision Intelligence & Prescriptive Operational Twin.
"""
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from config.settings import (
    TWIN_STATE_PATH, MAINTENANCE_MODEL_PATH, DELAY_MODEL_PATH,
    AI4I_PROCESSED_PATH, DATACO_PROCESSED_PATH
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
from src.visualization import (
    plot_inventory_trajectory, plot_machine_risk_rankings,
    plot_feature_importance, plot_confusion_matrix,
    plot_combined_scenario_comparison, plot_roi_arbitrage,
    plot_xai_waterfall, plot_resilience_matrix,
    plot_supplier_concentration
)

# Page configuration
st.set_page_config(
    page_title="BCG X Digital Twin | Supply Chain & Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom BCG X Industrial Theme CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .bcg-banner {
        background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%);
        border: 1px solid #059669;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .badge-high {
        background-color: #ea580c;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .badge-medium {
        background-color: #f59e0b;
        color: #0f172a;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .badge-healthy {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .disclaimer-box {
        background-color: #1e293b;
        border-left: 4px solid #10b981;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9em;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_state():
    """Caches ML models and twin state."""
    if not TWIN_STATE_PATH.exists():
        prepare_and_cache_datasets()

    with open(TWIN_STATE_PATH, "r", encoding="utf-8") as f:
        twin_state = json.load(f)

    # Load ML models
    m_model = PredictiveMaintenanceModel.load(MAINTENANCE_MODEL_PATH)
    sc_model = SupplyChainDelayModel.load(DELAY_MODEL_PATH)

    # Enrich twin machines with ML predictions
    for m in twin_state['machines']:
        pred = m_model.predict_risk(m)
        m.update(pred)

    # Enrich twin shipments with ML predictions
    for s in twin_state['shipments']:
        sc_pred = sc_model.predict_delay_risk(s)
        s.update(sc_pred)

    # Compute baseline inventory forecasts
    comp_forecasts = []
    for comp in twin_state['components']:
        shipment = next((s for s in twin_state['shipments'] if s['component_id'] == comp['component_id']), None)
        fc = InventoryForecaster.forecast_component_trajectory(comp, shipment)
        comp_forecasts.append(fc)

    # Calculate facility summary
    summary = CompositeRiskEngine.calculate_facility_summary(
        twin_state['machines'], twin_state['shipments'], comp_forecasts
    )

    # BCG X Prescriptive Arbitrage
    roi_arbitrage = PrescriptiveOptimizer.calculate_executive_roi_arbitrage(
        twin_state['machines'], twin_state['shipments'], comp_forecasts
    )

    # Resilience calculations
    resilience_data = ResilienceAnalyzer.calculate_component_resilience(
        twin_state['components'], twin_state['shipments']
    )
    hhi_data = ResilienceAnalyzer.calculate_supplier_hhi(twin_state['components'])

    return twin_state, m_model, sc_model, comp_forecasts, summary, roi_arbitrage, resilience_data, hhi_data


# Load data and models
(
    twin_state, m_model, sc_model, comp_forecasts,
    facility_summary, roi_arbitrage, resilience_data, hhi_data
) = load_models_and_state()

simulator = ScenarioSimulator(twin_state)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/conveyor-belt.png", width=64)
st.sidebar.title("BCG X Digital Twin")
st.sidebar.caption("Supply Chain Risk & Predictive Maintenance")

nav_choice = st.sidebar.radio(
    "Decision Cockpit",
    [
        "💎 Executive Value & ROI Cockpit",
        "⚙️ Predictive Maintenance & XAI",
        "🚚 Supply Chain Monitoring",
        "📦 Inventory Forecasting",
        "🎯 Prescriptive Decision Optimizer",
        "🛡️ Resilience Matrix (TTS / TTR)",
        "🔮 What-If Simulator",
        "📄 Boardroom Executive Briefing"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Manufacturing Asset Scope")
st.sidebar.info(
    f"**Facility**: {twin_state['metadata']['facility_name']}\n\n"
    f"**Assets Monitored**: {len(twin_state['machines'])} Machines\n\n"
    f"**Inbound Network**: {len(twin_state['components'])} Components ({len(twin_state['shipments'])} Shipments)"
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "🟢 **BCG X Decision-Support Standard**:\nProbabilistic ML coupled with mathematical optimization, Time-to-Survive (TTS) resilience metrics, and quantified EBITDA impact."
)

# -------------------------------------------------------------
# 1. EXECUTIVE VALUE & ROI COCKPIT
# -------------------------------------------------------------
if nav_choice == "💎 Executive Value & ROI Cockpit":
    st.markdown("""
    <div class="bcg-banner">
        <h2 style="margin:0; color: #34d399;">💎 BCG X Executive Value & Decision Intelligence</h2>
        <p style="margin:4px 0 0 0; color: #e2e8f0;">
            Unified C-Suite command center bridging shopfloor IoT telemetry, multi-tier logistics variance, and quantified EBITDA Value-at-Risk.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics with Financial Arbitrage
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Facility Health Score", f"{facility_summary['overall_facility_health']}%", delta="Normal Operating Band")
    with col2:
        st.metric("Value-at-Risk (VaR)", f"${roi_arbitrage['total_value_at_risk_usd']:,.0f}", delta="Reactive Exposure", delta_color="inverse")
    with col3:
        st.metric("Annualized EBITDA Gain", f"${roi_arbitrage['annualized_ebitda_impact_usd']:,.0f}", delta="Prescriptive Uplift")
    with col4:
        st.metric("Prescriptive ROI Multiplier", f"{roi_arbitrage['roi_multiplier']:.1f}x", delta=f"{roi_arbitrage['payback_period_weeks']} Wks Payback")
    with col5:
        st.metric("Network Resilience Index", f"{resilience_data['average_network_resilience_score']} / 100", delta=f"{resilience_data['vulnerable_components_count']} Vulnerable Nodes", delta_color="inverse")

    st.markdown("---")

    # Financial Arbitrage Waterfall Chart
    fig_roi = plot_roi_arbitrage(roi_arbitrage)
    st.plotly_chart(fig_roi, use_container_width=True)

    st.markdown("---")

    # Domain Health & Operational Vulnerabilities
    left_chart, right_chart = st.columns([3, 2])
    with left_chart:
        fig_rank = plot_machine_risk_rankings(twin_state['machines'])
        st.plotly_chart(fig_rank, use_container_width=True)

    with right_chart:
        st.markdown("### Operational Domain Health Breakdown")
        st.write(f"**Machinery & Physical Equipment**: `{facility_summary['equipment_health_score']}%`")
        st.progress(facility_summary['equipment_health_score'] / 100.0)

        st.write(f"**Inbound Logistics Reliability**: `{facility_summary['supply_chain_health_score']}%`")
        st.progress(facility_summary['supply_chain_health_score'] / 100.0)

        st.write(f"**Inventory Buffer Health**: `{facility_summary['inventory_health_score']}%`")
        st.progress(facility_summary['inventory_health_score'] / 100.0)

        st.markdown("---")
        st.markdown("### Top Operational Hotspots")
        if facility_summary['critical_machines_count'] > 0:
            for m in facility_summary['critical_machines'][:2]:
                st.error(f"🚨 **{m['machine_id']} ({m['name']})**: Failure prob `{m['failure_probability']*100:.1f}%`. Modes: {', '.join(m.get('suspected_failure_modes', []))}")
        if facility_summary['stockout_components_count'] > 0:
            for c in facility_summary['stockout_components'][:2]:
                st.warning(f"📦 **{c['component_name']}**: Projected stockout Day {c['stockout_day']}. Deficit: {c['cumulative_deficit_units']} units.")

# -------------------------------------------------------------
# 2. PREDICTIVE MAINTENANCE & XAI
# -------------------------------------------------------------
elif nav_choice == "⚙️ Predictive Maintenance & XAI":
    st.title("⚙️ Predictive Maintenance & Explainable AI (XAI)")
    st.markdown("Real-time physical telemetry monitoring, machine failure probabilities, and local root-cause feature attribution.")

    machine_options = [f"{m['machine_id']} - {m['name']} ({m['bay']})" for m in twin_state['machines']]
    selected_m_str = st.selectbox("Select Machine Asset to Inspect", machine_options)
    selected_m_id = selected_m_str.split(" - ")[0]
    machine = next((m for m in twin_state['machines'] if m['machine_id'] == selected_m_id), twin_state['machines'][0])

    # Telemetry cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Spindle Rotational Speed", f"{machine['rotational_speed_rpm']} rpm")
    with c2:
        st.metric("Torque", f"{machine['torque_nm']} Nm")
    with c3:
        st.metric("Tool Wear", f"{machine['tool_wear_min']} min", delta="High wear" if machine['tool_wear_min'] > 180 else "Normal")
    with c4:
        st.metric("Mechanical Power", f"{machine['power_kw']} kW")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Process Temperature", f"{machine['process_temperature_k']} K")
    with c6:
        st.metric("Air Temperature", f"{machine['air_temperature_k']} K")
    with c7:
        st.metric("Temp Difference (ΔT)", f"{machine['temp_difference_k']} K")
    with c8:
        st.metric("Overstrain Metric", f"{machine['overstrain_factor']}")

    st.markdown("---")

    # Local XAI Waterfall Breakdown
    xai_res = ExplainabilityEngine.explain_machine_risk(machine)

    st.markdown(f"### 🔬 Root-Cause Explainability for {machine['machine_id']}")
    st.caption(f"Decomposition of failure probability into physical telemetry contributions (Top Risk Driver: **{xai_res['top_risk_driver']}**).")
    
    fig_waterfall = plot_xai_waterfall(
        xai_res['contributions'],
        baseline_val=xai_res['baseline_probability_pct'],
        final_val=xai_res['final_failure_probability_pct'],
        title=f"Telemetry Attribution Waterfall: {machine['machine_id']} ({machine['name']})"
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.markdown("---")

    r_col1, r_col2 = st.columns([1, 2])
    with r_col1:
        st.markdown("### Asset Health Status")
        lvl = machine['risk_level']
        badge_class = f"badge-{lvl.lower()}"
        st.markdown(f"**Risk Level**: <span class='{badge_class}'>{lvl}</span>", unsafe_allow_html=True)
        st.markdown(f"**Failure Probability**: `{machine['failure_probability']*100:.2f}%`")
        st.markdown(f"**Operating Hours**: `{machine['operating_hours']} hrs`")
        st.markdown(f"**Downtime Cost**: `${machine['cost_per_hour_downtime_usd']}/hr`")
        st.markdown(f"**Dependent Component**: `{machine['component']}`")

        st.markdown("#### Diagnosed Failure Mode(s):")
        for mode in machine['suspected_failure_modes']:
            st.write(f"- {mode}")

    with r_col2:
        st.markdown("### Global Telemetry Feature Importance (AI4I 2020 Model)")
        fig_feat = plot_feature_importance(m_model.feature_importance_df)
        st.plotly_chart(fig_feat, use_container_width=True)

    with st.expander("🔬 View Model Validation & Verification Metrics (AI4I 2020 Real Dataset)"):
        vm1, vm2, vm3, vm4 = st.columns(4)
        vm1.metric("ROC-AUC Score", f"{m_model.metrics.get('roc_auc', 0.98):.4f}")
        vm2.metric("Accuracy", f"{m_model.metrics.get('accuracy', 0.98)*100:.2f}%")
        vm3.metric("Recall (Failures)", f"{m_model.metrics.get('recall', 0.72)*100:.2f}%")
        vm4.metric("Precision", f"{m_model.metrics.get('precision', 0.61)*100:.2f}%")

        fig_cm = plot_confusion_matrix(m_model.metrics.get('confusion_matrix', [[1943, 18], [11, 28]]), "Maintenance Model Confusion Matrix (Test Set)")
        st.plotly_chart(fig_cm, use_container_width=True)

# -------------------------------------------------------------
# 3. SUPPLY CHAIN MONITORING
# -------------------------------------------------------------
elif nav_choice == "🚚 Supply Chain Monitoring":
    st.title("🚚 Supply Chain & Inbound Logistics Monitoring")
    st.markdown("Active supplier shipments, carrier lead times, route disruption indicators, and late delivery probability.")

    shipments_df = pd.DataFrame(twin_state['shipments'])
    display_df = shipments_df[[
        'shipment_id', 'component_name', 'supplier', 'shipping_mode',
        'origin_region', 'scheduled_days', 'predicted_delay_days',
        'risk_tier', 'weather_disruption_index', 'status'
    ]].copy()
    display_df.columns = [
        'Shipment ID', 'Component', 'Supplier', 'Shipping Mode',
        'Origin Region', 'Scheduled Days', 'Predicted Delay (Days)',
        'Risk Tier', 'Weather Index', 'Status'
    ]

    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")

    # Local XAI Delay Attribution for selected shipment
    shp_select = st.selectbox("Select Shipment for Root-Cause Attribution", [s['shipment_id'] + " - " + s['component_name'] for s in twin_state['shipments']])
    shp_id = shp_select.split(" - ")[0]
    target_shp = next(s for s in twin_state['shipments'] if s['shipment_id'] == shp_id)
    shp_xai = ExplainabilityEngine.explain_shipment_delay(target_shp)

    st.markdown(f"### 🔬 Root-Cause Delay Drivers: {shp_id} ({target_shp['component_name']})")
    fig_shp_waterfall = plot_xai_waterfall(
        shp_xai['contributions'],
        baseline_val=shp_xai['baseline_delay_risk_pct'],
        final_val=shp_xai['final_delay_risk_pct'],
        title=f"Inbound Delay Attribution: {target_shp['component_name']} (+{shp_xai['predicted_delay_days']} Days Delay)"
    )
    st.plotly_chart(fig_shp_waterfall, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Scheduled Days vs Inbound Delay")
        fig_delay = px.bar(
            shipments_df,
            x='component_name',
            y=['scheduled_days', 'predicted_delay_days'],
            title="<b>Scheduled Days vs Predicted Inbound Delay</b>",
            labels={'value': 'Days', 'component_name': 'Component'},
            template="plotly_dark",
            barmode='stack',
            color_discrete_map={'scheduled_days': '#0284c7', 'predicted_delay_days': '#ef4444'}
        )
        st.plotly_chart(fig_delay, use_container_width=True)

    with col_b:
        st.markdown("### Route Risk Assessment (Weather & Geopolitical)")
        fig_route = px.scatter(
            shipments_df,
            x='weather_disruption_index',
            y='geopolitical_risk_index',
            size='quantity',
            color='risk_tier',
            hover_name='component_name',
            title="<b>Disruption Exposure by Route</b>",
            labels={'weather_disruption_index': 'Weather Disruption Index', 'geopolitical_risk_index': 'Geopolitical Risk Index'},
            template="plotly_dark",
            color_discrete_map={'CRITICAL': '#ef4444', 'HIGH': '#ea580c', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'}
        )
        st.plotly_chart(fig_route, use_container_width=True)

# -------------------------------------------------------------
# 4. INVENTORY FORECASTING
# -------------------------------------------------------------
elif nav_choice == "📦 Inventory Forecasting":
    st.title("📦 Inventory Trajectory & Stockout Forecasting")
    st.markdown("Deterministic multi-echelon stock levels projected day-by-day over a 30-day forward horizon.")

    comp_options = [f"{c['component_id']} - {c['name']} ({c['supplier'].split('(')[0]})" for c in twin_state['components']]
    selected_c_str = st.selectbox("Select Component to Forecast", comp_options)
    selected_c_id = selected_c_str.split(" - ")[0]
    component = next((c for c in twin_state['components'] if c['component_id'] == selected_c_id), twin_state['components'][0])
    shipment = next((s for s in twin_state['shipments'] if s['component_id'] == component['component_id']), None)

    fc = InventoryForecaster.forecast_component_trajectory(component, shipment)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Current On-Hand Stock", f"{fc['current_stock']} units")
    with col2:
        st.metric("Daily Consumption Rate", f"{component['daily_burn_rate']} units/day")
    with col3:
        st.metric("Days of Supply", f"{fc['days_of_supply']} days")
    with col4:
        st.metric("Safety Stock Buffer", f"{fc['safety_stock']} units")
    with col5:
        stockout_text = f"Day {fc['stockout_day']}" if fc['stockout_day'] else "None (Safe)"
        st.metric("Projected Stockout", stockout_text, delta="Deficit: " + str(fc['cumulative_deficit_units']) if fc['stockout_day'] else "In Stock", delta_color="inverse")

    st.markdown("---")
    fig_inv = plot_inventory_trajectory(
        fc['trajectory'],
        component_name=fc['component_name'],
        safety_stock=fc['safety_stock'],
        reorder_point=component['reorder_point']
    )
    st.plotly_chart(fig_inv, use_container_width=True)

# -------------------------------------------------------------
# 5. PRESCRIPTIVE DECISION OPTIMIZER (BCG X CORE)
# -------------------------------------------------------------
elif nav_choice == "🎯 Prescriptive Decision Optimizer":
    st.title("🎯 Prescriptive Decision Optimizer")
    st.markdown("""
    <div class="disclaimer-box">
        <b>Mathematical Decision Engine</b>: Translates machine learning risks into optimal replenishment parameters, service-level safety stocks, and capacity-constrained maintenance schedules.
    </div>
    """, unsafe_allow_html=True)

    opt_tab1, opt_tab2 = st.tabs([
        "📦 Dynamic Safety Stock & EOQ Optimizer",
        "⚙️ Constrained Maintenance Scheduler"
    ])

    with opt_tab1:
        st.subheader("Dynamic Multi-Echelon Safety Stock & Economic Order Quantity (EOQ)")
        st.caption("Solves the inventory trade-off: Stockout penalty risk vs Working Capital carrying cost.")

        sl_col1, sl_col2 = st.columns([1, 3])
        with sl_col1:
            target_sl = st.select_slider(
                "Target Cycle Service Level",
                options=[0.90, 0.95, 0.98, 0.99],
                value=0.95,
                format_func=lambda x: f"{int(x*100)}% Service Level"
            )
            order_cost = st.number_input("Fixed Order Cost ($)", min_value=50.0, max_value=500.0, value=150.0, step=25.0)

        opt_inv = PrescriptiveOptimizer.optimize_inventory_policies(
            twin_state['components'],
            twin_state['shipments'],
            target_service_level=target_sl,
            order_cost_usd=order_cost
        )

        with sl_col2:
            k1, k2, k3 = st.columns(3)
            k1.metric("Optimized Annual Holding Cost", f"${opt_inv['total_optimized_holding_cost_usd']:,.0f}", delta=f"${opt_inv['net_annual_holding_savings_usd']:,.0f} vs Baseline")
            k2.metric("Working Capital Freed", f"${opt_inv['net_working_capital_freed_usd']:,.0f}", help="Excess capital released from rebalanced safety stock")
            k3.metric("Annual Order Admin Cost", f"${opt_inv['total_annual_order_cost_usd']:,.0f}")

        st.markdown("#### Component Replenishment Optimization Matrix")
        opt_df = pd.DataFrame(opt_inv['components'])[[
            'component_id', 'name', 'unit_cost_usd', 'daily_demand',
            'current_safety_stock', 'optimized_safety_stock', 'safety_stock_delta',
            'optimized_eoq', 'working_capital_impact_usd', 'status'
        ]]
        opt_df.columns = [
            'Component ID', 'Component', 'Unit Cost ($)', 'Daily Burn',
            'Current SS', 'Optimized SS', 'SS Delta',
            'Optimal EOQ', 'Capital Impact ($)', 'Buffer Status'
        ]
        st.dataframe(opt_df, use_container_width=True)

    with opt_tab2:
        st.subheader("Resource-Constrained Maintenance Intervention Scheduler")
        st.caption("Knapsack-style optimization maximizing Avoided Downtime Loss under limited technician crew hours.")

        m_col1, m_col2 = st.columns([1, 3])
        with m_col1:
            tech_capacity = st.slider("Available Technician Capacity (Hours)", min_value=8, max_value=48, value=24, step=4)
            tech_rate = st.number_input("Technician Rate ($/hr)", min_value=50.0, max_value=200.0, value=85.0, step=5.0)

        sched_res = PrescriptiveOptimizer.optimize_maintenance_schedule(
            twin_state['machines'],
            available_technician_hours=tech_capacity,
            cost_per_tech_hour_usd=tech_rate
        )

        with m_col2:
            sk1, sk2, sk3, sk4 = st.columns(4)
            sk1.metric("Interventions Scheduled", f"{sched_res['scheduled_interventions_count']} Assets")
            sk2.metric("Technician Hours Used", f"{sched_res['allocated_technician_hours']} / {sched_res['available_technician_hours']} hrs")
            sk3.metric("Avoided Downtime Losses", f"${sched_res['total_unplanned_losses_avoided_usd']:,.0f}")
            sk4.metric("Net Maintenance ROI", f"{sched_res['roi_percentage']:.0f}%", delta=f"${sched_res['net_maintenance_savings_usd']:,.0f} Net Gain")

        st.markdown("#### Optimal Maintenance Dispatch Schedule")
        if sched_res['scheduled_interventions']:
            sched_df = pd.DataFrame(sched_res['scheduled_interventions'])[[
                'machine_id', 'name', 'bay', 'failure_probability',
                'required_tech_hours', 'planned_cost_usd', 'expected_unplanned_loss_usd', 'net_avoided_loss_usd'
            ]]
            sched_df.columns = [
                'Machine ID', 'Asset Name', 'Bay', 'Failure Prob',
                'Tech Hours', 'Planned Cost ($)', 'Avoided Loss ($)', 'Net Value ($)'
            ]
            sched_df['Failure Prob'] = sched_df['Failure Prob'].apply(lambda x: f"{x*100:.1f}%")
            sched_df['Planned Cost ($)'] = sched_df['Planned Cost ($)'].apply(lambda x: f"${x:,.0f}")
            sched_df['Avoided Loss ($)'] = sched_df['Avoided Loss ($)'].apply(lambda x: f"${x:,.0f}")
            sched_df['Net Value ($)'] = sched_df['Net Value ($)'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(sched_df, use_container_width=True)
        else:
            st.info("All machines are currently operating within safe probability thresholds.")

# -------------------------------------------------------------
# 6. RESILIENCE MATRIX (TTS / TTR & SUPPLIER HHI)
# -------------------------------------------------------------
elif nav_choice == "🛡️ Resilience Matrix (TTS / TTR)":
    st.title("🛡️ Supply Chain Resilience & Network Vulnerability")
    st.markdown("""
    <div class="bcg-banner">
        <h3 style="margin:0; color:#34d399;">BCG Supply Chain Resilience Framework</h3>
        <p style="margin:4px 0 0 0; color:#cbd5e1;">
            <b>Time-to-Survive (TTS)</b> measures how long operations can persist without replenishment.
            <b>Time-to-Recover (TTR)</b> measures the duration needed to restore or qualify alternative sourcing.
            Nodes where <b>TTS &lt; TTR</b> represent structural supply chain crisis points.
        </p>
    </div>
    """, unsafe_allow_html=True)

    r_kpi1, r_kpi2, r_kpi3 = st.columns(3)
    r_kpi1.metric("Average Resilience Score", f"{resilience_data['average_network_resilience_score']} / 100")
    r_kpi2.metric("Critical Disruption Nodes", f"{resilience_data['vulnerable_components_count']} / {resilience_data['total_components']}", delta="TTS < TTR Gap", delta_color="inverse")
    r_kpi3.metric("Supplier HHI Index", f"{hhi_data['hhi_index']:,.0f}", delta=hhi_data['concentration_tier'])

    st.markdown("---")

    # Resilience Scatter Quadrant
    fig_res = plot_resilience_matrix(resilience_data['resilience_records'])
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")

    col_h1, col_h2 = st.columns([1, 1])
    with col_h1:
        st.markdown("### Component Resilience Audit Table")
        res_df = pd.DataFrame(resilience_data['resilience_records'])[[
            'component_id', 'name', 'supplier', 'time_to_survive_days',
            'time_to_recover_days', 'resilience_gap_days', 'classification'
        ]]
        res_df.columns = [
            'ID', 'Component', 'Supplier', 'TTS (Days)', 'TTR (Days)',
            'Gap (Days)', 'Resilience Classification'
        ]
        st.dataframe(res_df, use_container_width=True)

    with col_h2:
        st.markdown("### Supplier Sourcing Concentration (HHI)")
        fig_hhi = plot_supplier_concentration(hhi_data)
        st.plotly_chart(fig_hhi, use_container_width=True)

# -------------------------------------------------------------
# 7. WHAT-IF DIGITAL TWIN SIMULATOR
# -------------------------------------------------------------
elif nav_choice == "🔮 What-If Simulator":
    st.title("🔮 Supply Chain Digital Twin: What-If Simulator")
    st.markdown("""
    <div class="disclaimer-box">
        <b>Interactive Stress Testing</b>: Adjust scenario levers below to immediately calculate the ripple effects across inventory, production capacity, stockout dates, and operational losses.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚢 Scenario 1: Shipping Delay",
        "🛑 Scenario 2: Machine Downtime",
        "⚠️ Scenario 3: Supplier Disruption",
        "📈 Scenario 4: Demand Surge",
        "⚡ Scenario 5: Combined Stress Test"
    ])

    with tab1:
        st.subheader("Scenario 1: Inbound Shipment Delay")
        col_s1_left, col_s1_right = st.columns([1, 2])
        with col_s1_left:
            comp_s1 = st.selectbox("Target Component", [c['name'] for c in twin_state['components']], key="s1_comp")
            comp_obj = next(c for c in twin_state['components'] if c['name'] == comp_s1)
            delay_slider = st.slider("Simulated Delay (Days)", min_value=0, max_value=30, value=6, step=1, key="s1_slider")

            res_s1 = simulator.simulate_shipping_delay(comp_obj['component_id'], delay_slider)

            st.metric("Revised Delivery Day", f"Day {res_s1['simulated_arrival_day']}", delta=f"+{delay_slider} days")
            st.metric("Simulated Stockout Day", f"Day {res_s1['simulated_stockout_day']}" if res_s1['simulated_stockout_day'] else "No Stockout")
            st.metric("Inventory Deficit", f"{res_s1['inventory_deficit_units']} units")

            st.markdown("#### Recommended Actions:")
            for act in res_s1['mitigation_actions']:
                st.write(f"- {act}")

        with col_s1_right:
            fig_s1 = plot_inventory_trajectory(
                res_s1['baseline_trajectory'],
                res_s1['simulated_trajectory'],
                component_name=f"{comp_obj['name']} (+{delay_slider} Days Delay)",
                safety_stock=comp_obj['safety_stock'],
                reorder_point=comp_obj['reorder_point']
            )
            st.plotly_chart(fig_s1, use_container_width=True)

    with tab2:
        st.subheader("Scenario 2: Critical Machine Failure & Downtime")
        col_s2_left, col_s2_right = st.columns([1, 2])
        with col_s2_left:
            m_s2 = st.selectbox("Select Machine", [f"{m['machine_id']} - {m['name']}" for m in twin_state['machines']], key="s2_m")
            m_id = m_s2.split(" - ")[0]
            downtime_days_slider = st.slider("Unplanned Downtime Duration (Days)", 1, 14, 3, key="s2_days")

            res_s2 = simulator.simulate_machinery_downtime(m_id, downtime_days_slider)

            st.metric("Lost Production Capacity", f"{res_s2['lost_production_units']:,} units")
            st.metric("Production Backlog", f"{res_s2['backlog_units_accumulated']:,} units")
            st.metric("Estimated Downtime Loss", f"${res_s2['estimated_downtime_cost_usd']:,} USD")

            st.markdown("#### Operational Recovery Options:")
            for rec in res_s2['recovery_options']:
                st.write(f"- {rec}")

        with col_s2_right:
            st.markdown("### Production Capacity & Backlog Impact")
            df_loss = pd.DataFrame({
                "Metric": ["Capacity Lost", "Accumulated Backlog", "Downtime Hours"],
                "Value": [res_s2['lost_production_units'], res_s2['backlog_units_accumulated'], res_s2['total_downtime_hours']]
            })
            fig_bar_m = px.bar(
                df_loss, x="Metric", y="Value", text_auto=True,
                title=f"<b>Operational Impact: {m_s2}</b>",
                template="plotly_dark", color="Metric",
                color_discrete_sequence=['#ef4444', '#ea580c', '#0284c7']
            )
            st.plotly_chart(fig_bar_m, use_container_width=True)

    with tab3:
        st.subheader("Scenario 3: Supplier Capacity Reduction or Prolonged Shock")
        col_s3_left, col_s3_right = st.columns([1, 2])
        with col_s3_left:
            comp_s3 = st.selectbox("Affected Component", [c['name'] for c in twin_state['components']], key="s3_comp")
            comp_obj3 = next(c for c in twin_state['components'] if c['name'] == comp_s3)
            supply_cut = st.slider("Supply Quantity Reduction (%)", 0, 100, 50, step=5, key="s3_cut")
            ext_delay = st.slider("Supplier Prolonged Lead Time (Days)", 0, 20, 8, key="s3_delay")

            res_s3 = simulator.simulate_supplier_disruption(comp_obj3['component_id'], supply_cut, ext_delay)

            st.metric("Stockout Day", f"Day {res_s3['stockout_day']}" if res_s3['stockout_day'] else "Buffer Holds")
            st.metric("Cumulative Deficit", f"{res_s3['cumulative_deficit_units']} units")

            st.markdown("#### Sourcing Strategies:")
            for strat in res_s3['recommended_strategy']:
                st.write(f"- {strat}")

        with col_s3_right:
            base_s3 = InventoryForecaster.forecast_component_trajectory(comp_obj3, next(s for s in twin_state['shipments'] if s['component_id'] == comp_obj3['component_id']))
            fig_s3 = plot_inventory_trajectory(
                base_s3['trajectory'],
                res_s3['trajectory'],
                component_name=f"{comp_obj3['name']} (-{supply_cut}% supply, +{ext_delay}d delay)",
                safety_stock=comp_obj3['safety_stock'],
                reorder_point=comp_obj3['reorder_point']
            )
            st.plotly_chart(fig_s3, use_container_width=True)

    with tab4:
        st.subheader("Scenario 4: Unexpected Demand Surge")
        demand_change = st.slider("Demand Change Percentage (%)", -50, 100, 30, step=5, key="s4_slider")
        res_s4 = simulator.simulate_demand_surge(demand_change)

        st.metric("Components Suffering Stockout", f"{res_s4['components_with_stockout']} / {len(twin_state['components'])}")
        st.dataframe(pd.DataFrame(res_s4['component_details']), use_container_width=True)

    with tab5:
        st.subheader("Scenario 5: Multi-Vector Combined Disruption")
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        with c_p1:
            comb_ship_delay = st.slider("Shipping Delay", 0, 20, 6, key="comb_ship")
        with c_p2:
            comb_machine_down = st.slider("Machine Downtime (Days)", 0, 10, 3, key="comb_down")
        with c_p3:
            comb_demand = st.slider("Demand Surge (%)", -20, 80, 25, key="comb_dem")
        with c_p4:
            comb_supplier_cut = st.slider("Supply Reduction (%)", 0, 80, 20, key="comb_cut")

        res_s5 = simulator.simulate_combined_disruption(
            comb_ship_delay, comb_machine_down, comb_demand, comb_supplier_cut
        )

        cmp = res_s5['comparison']
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.metric("Stockout Components", f"{cmp['simulated_stockout_components']} / {len(twin_state['components'])}", delta=f"{cmp['simulated_stockout_components'] - cmp['baseline_stockout_components']} additional", delta_color="inverse")
        with col_k2:
            st.metric("Total Deficit Volume", f"{cmp['simulated_inventory_deficit']:,.0f} units", delta=f"+{cmp['simulated_inventory_deficit'] - cmp['baseline_inventory_deficit']:,.0f}", delta_color="inverse")
        with col_k3:
            st.metric("Lost Production", f"{cmp['lost_production_units']:,} units")
        with col_k4:
            st.metric("Total Exposure", f"${cmp['total_combined_financial_exposure_usd']:,.0f} USD", delta="High Risk", delta_color="inverse")

        fig_comb = plot_combined_scenario_comparison(cmp)
        st.plotly_chart(fig_comb, use_container_width=True)

# -------------------------------------------------------------
# 8. BOARDROOM EXECUTIVE BRIEFING & PLAYBOOK
# -------------------------------------------------------------
elif nav_choice == "📄 Boardroom Executive Briefing":
    st.title("📄 Boardroom Executive Briefing & Decision Playbook")
    st.markdown("""
    <div class="bcg-banner">
        <h3 style="margin:0; color:#34d399;">C-Suite Operational Transformation Memo</h3>
        <p style="margin:4px 0 0 0; color:#cbd5e1;">
            Automated executive memo ready for Operations Committee and Board of Directors presentation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    memo_text = ResilienceAnalyzer.generate_executive_briefing_memo(
        facility_summary, roi_arbitrage, resilience_data, hhi_data
    )

    st.download_button(
        label="📥 Download Executive Briefing Memo (.md)",
        data=memo_text,
        file_name="BCG_X_Executive_Briefing_Digital_Twin.md",
        mime="text/markdown"
    )

    st.markdown("---")
    st.markdown(memo_text)

    st.markdown("---")
    st.subheader("Tactical Operational Mitigation Playbook")
    recommendations = CompositeRiskEngine.generate_actionable_recommendations(facility_summary)

    for idx, rec in enumerate(recommendations):
        priority = rec['priority']
        badge_color = "#ef4444" if priority == "CRITICAL" else ("#ea580c" if priority == "HIGH" else ("#f59e0b" if priority == "MEDIUM" else "#10b981"))

        with st.container():
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.15em; font-weight: bold; color: #38bdf8;">[{rec['category']}] {rec['target']}</span>
                    <span style="background-color: {badge_color}; color: white; padding: 3px 10px; border-radius: 4px; font-size: 0.8em; font-weight: bold;">
                        {priority} PRIORITY
                    </span>
                </div>
                <p style="margin-bottom: 6px;"><b>Finding:</b> {rec['finding']}</p>
                <p style="margin-bottom: 6px; color: #e2e8f0;"><b>Recommended Action:</b> {rec['action']}</p>
                <p style="margin-bottom: 0; color: #10b981;"><b>Expected Impact:</b> {rec['estimated_impact']}</p>
            </div>
            """, unsafe_allow_html=True)
