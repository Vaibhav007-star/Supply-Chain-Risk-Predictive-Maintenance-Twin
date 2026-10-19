"""
Visualization engine built with Plotly.
Provides executive KPI cards, telemetry diagnostics, inventory projection curves,
and digital-twin comparison visuals.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Industrial theme color palette
COLOR_PRIMARY = "#0284c7"     # Sky blue
COLOR_DANGER = "#ef4444"      # Crimson / critical
COLOR_WARNING = "#f59e0b"     # Amber
COLOR_SUCCESS = "#10b981"     # Emerald
COLOR_NEUTRAL = "#64748b"     # Slate
COLOR_DARK_BG = "#0f172a"     # Navy dark
COLOR_CARD_BG = "#1e293b"     # Slate card


def plot_inventory_trajectory(
    baseline_df: pd.DataFrame,
    simulated_df: pd.DataFrame = None,
    component_name: str = "Component",
    safety_stock: float = 100,
    reorder_point: float = 200
) -> go.Figure:
    """
    Renders 30-day inventory projection trajectory comparing baseline with simulation.
    """
    fig = go.Figure()

    # Safety stock zone
    fig.add_trace(go.Scatter(
        x=baseline_df['day'],
        y=baseline_df['safety_stock_threshold'],
        mode='lines',
        line=dict(color=COLOR_WARNING, dash='dash', width=2),
        name=f"Safety Stock ({safety_stock} units)"
    ))

    # Reorder point line
    fig.add_trace(go.Scatter(
        x=baseline_df['day'],
        y=baseline_df['reorder_point'],
        mode='lines',
        line=dict(color="#8b5cf6", dash='dot', width=2),
        name=f"Reorder Point ({reorder_point} units)"
    ))

    # Baseline trajectory
    fig.add_trace(go.Scatter(
        x=baseline_df['day'],
        y=baseline_df['inventory_level'],
        mode='lines+markers',
        line=dict(color=COLOR_PRIMARY, width=3),
        marker=dict(size=5),
        name="Baseline Projected Stock"
    ))

    # Simulated trajectory if provided
    if simulated_df is not None:
        fig.add_trace(go.Scatter(
            x=simulated_df['day'],
            y=simulated_df['inventory_level'],
            mode='lines+markers',
            line=dict(color=COLOR_DANGER, width=3, dash='solid'),
            marker=dict(size=6, symbol='diamond'),
            name="Simulated Scenario Stock"
        ))

    # Zero stock baseline (stockout floor)
    fig.add_hline(y=0, line_dash="solid", line_color="#dc2626", annotation_text="Stockout Threshold (0 units)")

    fig.update_layout(
        title=f"<b>30-Day Inventory Trajectory & Stockout Analysis: {component_name}</b>",
        xaxis_title="Simulation Day",
        yaxis_title="Physical On-Hand Inventory (Units)",
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_machine_risk_rankings(machines: List[Dict[str, Any]]) -> go.Figure:
    """
    Bar chart ranking machines by failure probability with risk tier coloring.
    """
    df = pd.DataFrame(machines)
    df = df.sort_values(by="failure_probability", ascending=True)

    color_map = {
        "CRITICAL": COLOR_DANGER,
        "HIGH": "#ea580c",
        "MEDIUM": COLOR_WARNING,
        "HEALTHY": COLOR_SUCCESS
    }
    colors = [color_map.get(lvl, COLOR_NEUTRAL) for lvl in df['risk_level']]

    fig = go.Figure(go.Bar(
        x=df['failure_probability'] * 100,
        y=df['machine_id'] + " - " + df['name'],
        orientation='h',
        marker=dict(color=colors, line=dict(color="#ffffff", width=0.5)),
        text=[f"{p*100:.1f}% ({lvl})" for p, lvl in zip(df['failure_probability'], df['risk_level'])],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>Machine Failure Probability & Risk Ranking</b>",
        xaxis_title="Failure Probability (%)",
        yaxis_title="Machine Asset",
        template="plotly_dark",
        margin=dict(l=40, r=60, t=50, b=40)
    )
    return fig


def plot_feature_importance(feature_df: pd.DataFrame) -> go.Figure:
    """
    Plots top model feature importances with percentage contributions.
    """
    df_sorted = feature_df.sort_values(by="importance", ascending=True).tail(8)

    fig = go.Figure(go.Bar(
        x=df_sorted['importance'] * 100,
        y=df_sorted['feature'],
        orientation='h',
        marker=dict(color="#38bdf8"),
        text=[f"{v*100:.1f}%" for v in df_sorted['importance']],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>Predictive Maintenance: Key Risk Drivers</b>",
        xaxis_title="Relative Feature Importance (%)",
        yaxis_title="Telemetry Feature",
        template="plotly_dark",
        margin=dict(l=40, r=50, t=50, b=40)
    )
    return fig


def plot_confusion_matrix(cm: List[List[int]], title: str = "Confusion Matrix") -> go.Figure:
    """
    Renders confusion matrix heatmap.
    """
    labels = ["Normal (0)", "Failure (1)"]
    z = cm

    fig = px.imshow(
        z,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Observations"),
        text_auto=True
    )

    fig.update_layout(
        title=f"<b>{title}</b>",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_combined_scenario_comparison(comparison_data: Dict[str, Any]) -> go.Figure:
    """
    Side-by-side grouped bar chart comparing baseline vs simulated scenario metrics.
    """
    metrics = ["Stockout Components", "Deficit Units (x100)", "Lost Capacity (x100 units)"]
    baseline_vals = [
        comparison_data.get('baseline_stockout_components', 0),
        round(comparison_data.get('baseline_inventory_deficit', 0) / 100, 1),
        0
    ]
    simulated_vals = [
        comparison_data.get('simulated_stockout_components', 0),
        round(comparison_data.get('simulated_inventory_deficit', 0) / 100, 1),
        round(comparison_data.get('lost_production_units', 0) / 100, 1)
    ]

    fig = go.Figure(data=[
        go.Bar(name='Baseline Operations', x=metrics, y=baseline_vals, marker_color=COLOR_PRIMARY),
        go.Bar(name='Simulated Disruption', x=metrics, y=simulated_vals, marker_color=COLOR_DANGER)
    ])

    fig.update_layout(
        barmode='group',
        title="<b>Baseline vs Simulated Disruption Impact Comparison</b>",
        yaxis_title="Magnitude",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_roi_arbitrage(roi_data: Dict[str, Any]) -> go.Figure:
    """
    Renders BCG executive value-capture waterfall / comparison chart.
    Contrasts Reactive Status Quo Losses against Prescriptive Mitigations & Net Savings.
    """
    categories = [
        "Reactive Downtime Loss",
        "Reactive Stockout Penalties",
        "Reactive Rush Freight",
        "Prescriptive Maint. Cost",
        "Prescriptive Buffer Cost",
        "Net Quarterly Savings"
    ]
    values = [
        roi_data.get('reactive_downtime_loss_usd', 0),
        roi_data.get('reactive_stockout_loss_usd', 0),
        roi_data.get('reactive_rush_logistics_usd', 0),
        -roi_data.get('prescriptive_maintenance_cost_usd', 0),
        -roi_data.get('prescriptive_buffer_cost_usd', 0),
        roi_data.get('net_quarterly_savings_usd', 0)
    ]
    colors = [COLOR_DANGER, COLOR_DANGER, "#ea580c", "#38bdf8", "#818cf8", COLOR_SUCCESS]

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"${abs(v):,.0f}" for v in values],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"<b>Financial Value-at-Risk Arbitrage: Reactive vs Prescriptive AI (ROI: {roi_data.get('roi_multiplier', 0):.1f}x)</b>",
        yaxis_title="USD ($)",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=60)
    )
    return fig


def plot_xai_waterfall(
    contributions: List[Dict[str, Any]],
    baseline_val: float,
    final_val: float,
    title: str = "Root Cause Feature Attribution"
) -> go.Figure:
    """
    Renders Plotly Waterfall chart visualizing exact feature risk attributions.
    """
    x_labels = ["Baseline Normal Risk"] + [c['factor'] for c in contributions] + ["Final Probability"]
    measure = ["absolute"] + ["relative"] * len(contributions) + ["total"]
    y_vals = [baseline_val] + [c.get('delta', c.get('delta_pct', 0)) for c in contributions] + [final_val]

    fig = go.Figure(go.Waterfall(
        name="Risk Drivers",
        orientation="v",
        measure=measure,
        x=x_labels,
        textposition="outside",
        text=[f"{v:+.1f}%" if m == "relative" else f"{v:.1f}%" for v, m in zip(y_vals, measure)],
        y=y_vals,
        connector={"line": {"color": "#64748b"}},
        decreasing={"marker": {"color": COLOR_SUCCESS}},
        increasing={"marker": {"color": COLOR_DANGER}},
        totals={"marker": {"color": "#e11d48"}}
    ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        yaxis_title="Risk Impact (Percentage Points)",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=70)
    )
    return fig


def plot_resilience_matrix(resilience_records: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders the BCG Time-to-Survive (TTS) vs Time-to-Recover (TTR) Resilience Matrix.
    Nodes with TTS < TTR fall in the critical disruption zone.
    """
    df = pd.DataFrame(resilience_records)

    fig = go.Figure()

    # Parity Line (TTS = TTR)
    max_val = max(df['time_to_survive_days'].max(), df['time_to_recover_days'].max()) + 4
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color="#94a3b8", dash='dash', width=2),
        name="Parity Boundary (TTS = TTR)"
    ))

    # Vulnerable Zone Shading
    fig.add_trace(go.Scatter(
        x=[0, max_val, 0],
        y=[0, max_val, max_val],
        fill='toself',
        fillcolor='rgba(239, 68, 68, 0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        name="Vulnerable Zone (TTS < TTR)",
        hoverinfo='skip'
    ))

    # Components scatter points
    for _, row in df.iterrows():
        is_vuln = row['resilience_gap_days'] < 0
        point_color = COLOR_DANGER if is_vuln else (COLOR_WARNING if row['resilience_gap_days'] <= 3 else COLOR_SUCCESS)
        fig.add_trace(go.Scatter(
            x=[row['time_to_survive_days']],
            y=[row['time_to_recover_days']],
            mode='markers+text',
            marker=dict(size=14, color=point_color, line=dict(color="#ffffff", width=1.5)),
            text=[row['name']],
            textposition="top center",
            name=row['name'],
            hovertemplate=(
                f"<b>{row['name']}</b><br>"
                f"Supplier: {row['supplier']}<br>"
                f"Time-to-Survive (TTS): {row['time_to_survive_days']} days<br>"
                f"Time-to-Recover (TTR): {row['time_to_recover_days']} days<br>"
                f"Resilience Gap: {row['resilience_gap_days']} days<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        title="<b>BCG Resilience Matrix: Time-to-Survive (TTS) vs Time-to-Recover (TTR)</b>",
        xaxis_title="Time-to-Survive [TTS] (Days until complete stockout)",
        yaxis_title="Time-to-Recover [TTR] (Days needed to restore supply)",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_supplier_concentration(hhi_data: Dict[str, Any]) -> go.Figure:
    """
    Renders supplier spend share and concentration breakdown.
    """
    df = pd.DataFrame(hhi_data.get('supplier_breakdown', []))

    fig = px.pie(
        df,
        names="supplier",
        values="annual_spend_usd",
        title=f"<b>Supplier Spend Concentration (HHI Index: {hhi_data.get('hhi_index', 0):,.0f} - {hhi_data.get('concentration_tier')})</b>",
        template="plotly_dark",
        hole=0.45,
        color_discrete_sequence=['#00875a', '#0284c7', '#38bdf8', '#818cf8', '#f59e0b', '#ef4444', '#ec4899']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig

