/**
 * BCG X Supply Chain Risk & Predictive Maintenance Twin
 * Client Application Logic & Plotly Visualizations
 */

// Global Application State
const AppState = {
    metadata: null,
    facilitySummary: null,
    roiArbitrage: null,
    machines: [],
    shipments: [],
    inventory: [],
    resilience: null,
    currentMachineId: null,
    currentShipmentId: null,
    currentCompId: null,
    memoData: null
};

const PLOTLY_THEME = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, sans-serif', color: '#94a3b8' },
    margin: { l: 40, r: 40, t: 40, b: 40 }
};

// Initialize Application on DOM Ready
document.addEventListener('DOMContentLoaded', async () => {
    setupNavigation();
    setupSimulatorSubtabs();
    setupEventListeners();
    await fetchAllData();
});

function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const pageTitle = document.getElementById('page-title');
    const pageDesc = document.getElementById('page-desc');

    const tabDescriptions = {
        'tab-overview': { title: 'Executive Value & ROI Cockpit', desc: 'Real-time telemetry, inbound logistics exposure, and quantified EBITDA impact.' },
        'tab-maintenance': { title: 'Predictive Maintenance & Explainable AI (XAI)', desc: 'Real-time telemetry diagnostics, Random Forest failure probabilities, and physics waterfalls.' },
        'tab-supply-chain': { title: 'Supply Chain Monitoring & Inbound Logistics', desc: 'Active supplier lead times, route disruption indicators, and late delivery probability.' },
        'tab-inventory': { title: 'Inventory Trajectory & Stockout Forecasting', desc: 'Deterministic 30-day forward projection curves with safety buffers and reorder points.' },
        'tab-prescriptive': { title: 'Prescriptive Decision Policy Optimizer', desc: 'Mathematical decision engine for dynamic safety stocks (EOQ) and constrained maintenance scheduling.' },
        'tab-resilience': { title: 'Supply Chain Resilience Matrix (TTS / TTR)', desc: 'BCG Time-to-Survive vs Time-to-Recover quadrant and Sourcing Concentration HHI.' },
        'tab-simulator': { title: 'What-If Digital Twin Disruption Simulator', desc: 'Deterministic scenario modeling for shipping delays, equipment downtime, and demand surges.' },
        'tab-memo': { title: 'Boardroom Executive Briefing & Decision Playbook', desc: 'Automated executive memo formatted in BCG consulting standard for C-suite review.' }
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(i => i.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            item.classList.add('active');
            const targetId = item.getAttribute('data-tab');
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');

            if (tabDescriptions[targetId]) {
                pageTitle.textContent = tabDescriptions[targetId].title;
                pageDesc.textContent = tabDescriptions[targetId].desc;
            }

            // Trigger window resize to re-render Plotly charts properly in new tab
            window.dispatchEvent(new Event('resize'));
        });
    });
}

function setupSimulatorSubtabs() {
    const subnavBtns = document.querySelectorAll('.subnav-btn');
    const simPanes = document.querySelectorAll('.sim-pane');

    subnavBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            subnavBtns.forEach(b => b.classList.remove('active'));
            simPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-subtab');
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');
            window.dispatchEvent(new Event('resize'));
        });
    });
}

function setupEventListeners() {
    document.getElementById('btn-refresh').addEventListener('click', fetchAllData);

    // Machine selector change
    document.getElementById('machine-selector').addEventListener('change', (e) => {
        loadMachineDetail(e.target.value);
    });

    // Shipment selector change
    document.getElementById('shipment-selector').addEventListener('change', (e) => {
        loadShipmentDetail(e.target.value);
    });

    // Inventory selector change
    document.getElementById('inventory-selector').addEventListener('change', (e) => {
        loadInventoryDetail(e.target.value);
    });

    // Prescriptive Service Level Slider
    const slSlider = document.getElementById('sl-slider');
    slSlider.addEventListener('input', (e) => {
        document.getElementById('sl-val').textContent = `${e.target.value}%`;
    });
    slSlider.addEventListener('change', (e) => {
        recomputeInventoryOptimization(parseFloat(e.target.value) / 100);
    });

    // Maintenance Tech Hours Slider
    const techSlider = document.getElementById('tech-hours-slider');
    techSlider.addEventListener('input', (e) => {
        document.getElementById('tech-val').textContent = `${e.target.value} Hours`;
    });
    techSlider.addEventListener('change', (e) => {
        recomputeMaintenanceOptimization(parseInt(e.target.value));
    });

    // Simulator Scenario 1
    const s1Delay = document.getElementById('sim1-delay');
    s1Delay.addEventListener('input', (e) => {
        document.getElementById('sim1-delay-val').textContent = `+${e.target.value} Days`;
    });
    s1Delay.addEventListener('change', runScenario1);
    document.getElementById('sim1-comp').addEventListener('change', runScenario1);

    // Simulator Scenario 2
    const s2Days = document.getElementById('sim2-days');
    s2Days.addEventListener('input', (e) => {
        document.getElementById('sim2-days-val').textContent = `${e.target.value} Days`;
    });
    s2Days.addEventListener('change', runScenario2);
    document.getElementById('sim2-machine').addEventListener('change', runScenario2);

    // Simulator Scenario 3
    const s3Cut = document.getElementById('sim3-cut');
    s3Cut.addEventListener('input', (e) => {
        document.getElementById('sim3-cut-val').textContent = `${e.target.value}%`;
    });
    s3Cut.addEventListener('change', runScenario3);
    const s3Delay = document.getElementById('sim3-delay');
    s3Delay.addEventListener('input', (e) => {
        document.getElementById('sim3-delay-val').textContent = `${e.target.value} Days`;
    });
    s3Delay.addEventListener('change', runScenario3);
    document.getElementById('sim3-comp').addEventListener('change', runScenario3);

    // Simulator Scenario 4
    const s4Surge = document.getElementById('sim4-surge');
    s4Surge.addEventListener('input', (e) => {
        document.getElementById('sim4-surge-val').textContent = `${e.target.value > 0 ? '+' : ''}${e.target.value}%`;
    });
    s4Surge.addEventListener('change', runScenario4);

    // Simulator Scenario 5
    ['sim5-ship', 'sim5-maint', 'sim5-demand', 'sim5-sup'].forEach(id => {
        const slider = document.getElementById(id);
        slider.addEventListener('input', (e) => {
            const valSpan = document.getElementById(`${id}-val`);
            if (id === 'sim5-ship' || id === 'sim5-maint') valSpan.textContent = `${e.target.value}d`;
            else valSpan.textContent = `${e.target.value > 0 ? '+' : ''}${e.target.value}%`;
        });
        slider.addEventListener('change', runScenario5);
    });

    // Download memo button
    document.getElementById('btn-download-memo').addEventListener('click', () => {
        if (!AppState.memoData) return;
        const blob = new Blob([AppState.memoData.memo_markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'BCG_X_Executive_Briefing_Digital_Twin.md';
        a.click();
        URL.revokeObjectURL(url);
    });
}

// -------------------------------------------------------------
// Data Fetching & Sync
// -------------------------------------------------------------
async function fetchAllData() {
    try {
        const [stateRes, machinesRes, shipmentsRes, invRes, resRes, memoRes] = await Promise.all([
            fetch('/api/state').then(r => r.json()),
            fetch('/api/machines').then(r => r.json()),
            fetch('/api/shipments').then(r => r.json()),
            fetch('/api/inventory').then(r => r.json()),
            fetch('/api/resilience').then(r => r.json()),
            fetch('/api/executive-memo').then(r => r.json())
        ]);

        AppState.metadata = stateRes.metadata;
        AppState.facilitySummary = stateRes.facility_summary;
        AppState.roiArbitrage = stateRes.roi_arbitrage;
        AppState.machines = machinesRes;
        AppState.shipments = shipmentsRes;
        AppState.inventory = invRes;
        AppState.resilience = resRes;
        AppState.memoData = memoRes;

        renderExecutiveOverview();
        populateDropdowns();
        renderShipmentsTable();
        renderResilienceSection();
        renderExecutiveMemo();

        // Initial detail loads
        if (AppState.machines.length > 0) loadMachineDetail(AppState.machines[0].machine_id);
        if (AppState.shipments.length > 0) loadShipmentDetail(AppState.shipments[0].shipment_id);
        if (AppState.inventory.length > 0) loadInventoryDetail(AppState.inventory[0].component_id);

        // Run default optimizations & simulations
        recomputeInventoryOptimization(0.95);
        recomputeMaintenanceOptimization(24);
        runScenario1();
        runScenario2();
        runScenario3();
        runScenario4();
        runScenario5();

    } catch (err) {
        console.error('Failed to load twin state:', err);
    }
}

// -------------------------------------------------------------
// 1. Executive Overview Rendering
// -------------------------------------------------------------
function renderExecutiveOverview() {
    const summary = AppState.facilitySummary;
    const roi = AppState.roiArbitrage;
    const res = AppState.resilience.component_resilience;

    document.getElementById('top-health-val').textContent = `${summary.overall_facility_health}%`;
    document.getElementById('kpi-health').textContent = `${summary.overall_facility_health}%`;
    document.getElementById('kpi-var').textContent = `$${Math.round(roi.total_value_at_risk_usd).toLocaleString()}`;
    document.getElementById('kpi-ebitda').textContent = `$${Math.round(roi.annualized_ebitda_impact_usd).toLocaleString()}`;
    document.getElementById('kpi-roi').textContent = `${roi.roi_multiplier.toFixed(1)}x`;
    document.getElementById('kpi-payback').textContent = `Payback: ${roi.payback_period_weeks} weeks`;
    document.getElementById('kpi-resilience').textContent = `${res.average_network_resilience_score} / 100`;
    document.getElementById('kpi-vuln-nodes').textContent = `${res.vulnerable_components_count} vulnerable components`;

    // Progress bars
    document.getElementById('domain-equip-pct').textContent = `${summary.equipment_health_score}%`;
    document.getElementById('bar-equip').style.width = `${summary.equipment_health_score}%`;
    document.getElementById('domain-sc-pct').textContent = `${summary.supply_chain_health_score}%`;
    document.getElementById('bar-sc').style.width = `${summary.supply_chain_health_score}%`;
    document.getElementById('domain-inv-pct').textContent = `${summary.inventory_health_score}%`;
    document.getElementById('bar-inv').style.width = `${summary.inventory_health_score}%`;

    // Hotspot list
    const hotspotContainer = document.getElementById('hotspot-list');
    hotspotContainer.innerHTML = '';
    if (summary.critical_machines_count > 0) {
        summary.critical_machines.forEach(m => {
            const div = document.createElement('div');
            div.className = 'hotspot-item';
            div.innerHTML = `<b>🚨 ${m.machine_id} (${m.name}):</b> Failure risk <b>${(m.failure_probability*100).toFixed(1)}%</b>. Tool wear: ${m.tool_wear_min} min.`;
            hotspotContainer.appendChild(div);
        });
    }
    if (summary.stockout_components_count > 0) {
        summary.stockout_components.forEach(c => {
            const div = document.createElement('div');
            div.className = 'hotspot-item';
            div.style.borderLeftColor = '#f59e0b';
            div.innerHTML = `<b>📦 ${c.component_name}:</b> Projected stockout on <b>Day ${c.stockout_day}</b> (Deficit: ${c.cumulative_deficit_units} units).`;
            hotspotContainer.appendChild(div);
        });
    }

    renderRoiArbitrageChart(roi);
    renderMachineRankingChart(AppState.machines);
}

function renderRoiArbitrageChart(roi) {
    const categories = [
        "Reactive Downtime Loss",
        "Reactive Stockout Loss",
        "Reactive Rush Freight",
        "Prescriptive Maint. Cost",
        "Prescriptive Buffer Cost",
        "Net Quarterly Savings"
    ];
    const values = [
        roi.reactive_downtime_loss_usd,
        roi.reactive_stockout_loss_usd,
        roi.reactive_rush_logistics_usd,
        -roi.prescriptive_maintenance_cost_usd,
        -roi.prescriptive_buffer_cost_usd,
        roi.net_quarterly_savings_usd
    ];
    const colors = ['#ef4444', '#ef4444', '#ea580c', '#38bdf8', '#818cf8', '#10b981'];

    const data = [{
        type: 'bar',
        x: categories,
        y: values,
        marker: { color: colors },
        text: values.map(v => `$${Math.abs(Math.round(v)).toLocaleString()}`),
        textposition: 'outside'
    }];

    const layout = {
        ...PLOTLY_THEME,
        yaxis: { title: 'USD ($)', gridcolor: 'rgba(255,255,255,0.06)' },
        xaxis: { tickangle: -15 }
    };

    Plotly.newPlot('chart-roi-arbitrage', data, layout, { responsive: true, displayModeBar: false });
}

function renderMachineRankingChart(machines) {
    const sorted = [...machines].sort((a, b) => a.failure_probability - b.failure_probability);
    const colorMap = {
        'CRITICAL': '#ef4444',
        'HIGH': '#ea580c',
        'MEDIUM': '#f59e0b',
        'HEALTHY': '#10b981'
    };

    const data = [{
        type: 'bar',
        orientation: 'h',
        x: sorted.map(m => m.failure_probability * 100),
        y: sorted.map(m => `${m.machine_id} - ${m.name}`),
        marker: { color: sorted.map(m => colorMap[m.risk_level] || '#64748b') },
        text: sorted.map(m => `${(m.failure_probability*100).toFixed(1)}% (${m.risk_level})`),
        textposition: 'outside'
    }];

    const layout = {
        ...PLOTLY_THEME,
        xaxis: { title: 'Failure Probability (%)', gridcolor: 'rgba(255,255,255,0.06)' },
        margin: { l: 150, r: 60, t: 20, b: 40 }
    };

    Plotly.newPlot('chart-machine-ranking', data, layout, { responsive: true, displayModeBar: false });
}

// -------------------------------------------------------------
// 2. Predictive Maintenance & XAI
// -------------------------------------------------------------
function populateDropdowns() {
    const machineSelect = document.getElementById('machine-selector');
    const sim2Machine = document.getElementById('sim2-machine');
    machineSelect.innerHTML = '';
    sim2Machine.innerHTML = '';

    AppState.machines.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.machine_id;
        opt.textContent = `${m.machine_id} - ${m.name} (${m.bay})`;
        machineSelect.appendChild(opt);

        const opt2 = opt.cloneNode(true);
        sim2Machine.appendChild(opt2);
    });

    const shpSelect = document.getElementById('shipment-selector');
    shpSelect.innerHTML = '';
    AppState.shipments.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.shipment_id;
        opt.textContent = `${s.shipment_id} - ${s.component_name} (${s.origin_region})`;
        shpSelect.appendChild(opt);
    });

    const invSelect = document.getElementById('inventory-selector');
    const sim1Comp = document.getElementById('sim1-comp');
    const sim3Comp = document.getElementById('sim3-comp');
    invSelect.innerHTML = '';
    sim1Comp.innerHTML = '';
    sim3Comp.innerHTML = '';

    AppState.inventory.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.component_id;
        opt.textContent = `${c.component_id} - ${c.component_name}`;
        invSelect.appendChild(opt);

        sim1Comp.appendChild(opt.cloneNode(true));
        sim3Comp.appendChild(opt.cloneNode(true));
    });
}

async function loadMachineDetail(machineId) {
    try {
        const res = await fetch(`/api/machine/${machineId}`).then(r => r.json());
        const m = res.machine;
        const xai = res.explainability;

        document.getElementById('tele-rpm').textContent = `${m.rotational_speed_rpm} rpm`;
        document.getElementById('tele-torque').textContent = `${m.torque_nm} Nm`;
        document.getElementById('tele-wear').textContent = `${m.tool_wear_min} min`;
        document.getElementById('tele-power').textContent = `${m.power_kw} kW`;
        document.getElementById('tele-ptemp').textContent = `${m.process_temperature_k} K`;
        document.getElementById('tele-atemp').textContent = `${m.air_temperature_k} K`;
        document.getElementById('tele-dtemp').textContent = `${m.temp_difference_k} K`;
        document.getElementById('tele-overstrain').textContent = m.overstrain_factor;

        const badge = document.getElementById('machine-status-badge');
        badge.className = `badge badge-${m.risk_level.toLowerCase()}`;
        badge.textContent = `${m.risk_level} (${(m.failure_probability*100).toFixed(1)}%)`;

        document.getElementById('xai-machine-caption').textContent = 
            `Top Risk Driver: ${xai.top_risk_driver} | Baseline Healthy: ${xai.baseline_probability_pct}% → Final Risk: ${xai.final_failure_probability_pct}%`;

        // Diagnostic box
        const diagBox = document.getElementById('diagnostic-details');
        diagBox.innerHTML = `
            <p><b>Operating Hours:</b> ${m.operating_hours} hrs</p>
            <p><b>Hourly Downtime Loss:</b> $${m.cost_per_hour_downtime_usd}/hr</p>
            <p><b>Dependent Manufacturing Component:</b> ${m.component}</p>
            <p class="mt-2"><b>Diagnosed Physics Failure Modes:</b></p>
            <ul class="action-list mt-1">
                ${m.suspected_failure_modes.map(mode => `<li>${mode}</li>`).join('')}
            </ul>
        `;

        renderXaiWaterfall('chart-machine-xai', xai.contributions, xai.baseline_probability_pct, xai.final_failure_probability_pct, `Telemetry Attribution Waterfall: ${m.machine_id}`);
        renderGlobalFeaturesChart();

    } catch (err) {
        console.error('Error loading machine details:', err);
    }
}

function renderXaiWaterfall(containerId, contributions, baseVal, finalVal, title) {
    const xLabels = ["Baseline Risk", ...contributions.map(c => c.factor), "Final Probability"];
    const measure = ["absolute", ...contributions.map(() => "relative"), "total"];
    const yVals = [baseVal, ...contributions.map(c => c.delta || c.delta_pct), finalVal];

    const data = [{
        type: "waterfall",
        orientation: "v",
        measure: measure,
        x: xLabels,
        y: yVals,
        text: yVals.map((v, i) => measure[i] === "relative" ? (v > 0 ? `+${v.toFixed(1)}%` : `${v.toFixed(1)}%`) : `${v.toFixed(1)}%`),
        textposition: "outside",
        connector: { line: { color: "#64748b" } },
        decreasing: { marker: { color: "#10b981" } },
        increasing: { marker: { color: "#ef4444" } },
        totals: { marker: { color: "#e11d48" } }
    }];

    const layout = {
        ...PLOTLY_THEME,
        title: { text: `<b>${title}</b>`, font: { size: 14, color: '#ffffff' } },
        yaxis: { title: "Risk Impact (%)", gridcolor: "rgba(255,255,255,0.06)" },
        xaxis: { tickangle: -15 }
    };

    Plotly.newPlot(containerId, data, layout, { responsive: true, displayModeBar: false });
}

function renderGlobalFeaturesChart() {
    const features = [
        { feature: "Rotational Speed", imp: 17.4 },
        { feature: "Mechanical Power", imp: 16.4 },
        { feature: "Torque", imp: 14.7 },
        { feature: "Tool Wear", imp: 12.0 },
        { feature: "Overstrain Metric", imp: 8.6 },
        { feature: "Temp Difference (ΔT)", imp: 7.8 },
        { feature: "Process Temperature", imp: 5.5 }
    ].reverse();

    const data = [{
        type: 'bar',
        orientation: 'h',
        x: features.map(f => f.imp),
        y: features.map(f => f.feature),
        marker: { color: '#38bdf8' },
        text: features.map(f => `${f.imp}%`),
        textposition: 'outside'
    }];

    const layout = {
        ...PLOTLY_THEME,
        xaxis: { title: 'Relative Importance (%)', gridcolor: 'rgba(255,255,255,0.06)' },
        margin: { l: 150, r: 40, t: 20, b: 40 }
    };

    Plotly.newPlot('chart-global-features', data, layout, { responsive: true, displayModeBar: false });
}

// -------------------------------------------------------------
// 3. Supply Chain Monitoring
// -------------------------------------------------------------
function renderShipmentsTable() {
    const tbody = document.querySelector('#table-shipments tbody');
    tbody.innerHTML = '';

    AppState.shipments.forEach(s => {
        const tr = document.createElement('tr');
        const badgeClass = `badge-${s.risk_tier.toLowerCase()}`;
        tr.innerHTML = `
            <td><b>${s.shipment_id}</b></td>
            <td>${s.component_name}</td>
            <td>${s.supplier}</td>
            <td>${s.shipping_mode}</td>
            <td>${s.origin_region}</td>
            <td>${s.scheduled_days} days</td>
            <td><b class="${s.predicted_delay_days > 0 ? 'text-rose' : 'text-emerald'}">+${s.predicted_delay_days} days</b></td>
            <td>${(s.weather_disruption_index*100).toFixed(0)}%</td>
            <td><span class="badge ${badgeClass}">${s.risk_tier}</span></td>
        `;
        tbody.appendChild(tr);
    });

    renderShipmentDelayBars();
    renderRouteRisksScatter();
}

async function loadShipmentDetail(shipmentId) {
    try {
        const res = await fetch(`/api/shipment/${shipmentId}`).then(r => r.json());
        const s = res.shipment;
        const xai = res.explainability;

        renderXaiWaterfall(
            'chart-shipment-xai',
            xai.contributions,
            xai.baseline_delay_risk_pct,
            xai.final_delay_risk_pct,
            `Inbound Delay Attribution: ${s.component_name} (+${s.predicted_delay_days} Days Delay)`
        );
    } catch (err) {
        console.error('Error loading shipment detail:', err);
    }
}

function renderShipmentDelayBars() {
    const data = [
        {
            name: 'Scheduled Days',
            type: 'bar',
            x: AppState.shipments.map(s => s.component_name),
            y: AppState.shipments.map(s => s.scheduled_days),
            marker: { color: '#0284c7' }
        },
        {
            name: 'Predicted Delay',
            type: 'bar',
            x: AppState.shipments.map(s => s.component_name),
            y: AppState.shipments.map(s => s.predicted_delay_days),
            marker: { color: '#ef4444' }
        }
    ];

    const layout = {
        ...PLOTLY_THEME,
        barmode: 'stack',
        title: { text: '<b>Scheduled vs Predicted Inbound Delay</b>', font: { size: 14, color: '#ffffff' } },
        yaxis: { title: 'Days', gridcolor: 'rgba(255,255,255,0.06)' },
        xaxis: { tickangle: -15 },
        legend: { orientation: 'h', y: 1.1 }
    };

    Plotly.newPlot('chart-shipment-delays', data, layout, { responsive: true, displayModeBar: false });
}

function renderRouteRisksScatter() {
    const colorMap = { 'CRITICAL': '#ef4444', 'HIGH': '#ea580c', 'MEDIUM': '#f59e0b', 'LOW': '#10b981' };

    const data = [{
        type: 'scatter',
        mode: 'markers+text',
        x: AppState.shipments.map(s => s.weather_disruption_index),
        y: AppState.shipments.map(s => s.geopolitical_risk_index),
        text: AppState.shipments.map(s => s.component_name),
        textposition: 'top center',
        marker: {
            size: AppState.shipments.map(s => Math.max(12, s.quantity / 25)),
            color: AppState.shipments.map(s => colorMap[s.risk_tier] || '#64748b'),
            line: { color: '#ffffff', width: 1.5 }
        }
    }];

    const layout = {
        ...PLOTLY_THEME,
        title: { text: '<b>Route Exposure: Weather vs Geopolitical Risk</b>', font: { size: 14, color: '#ffffff' } },
        xaxis: { title: 'Weather Disruption Index', gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Geopolitical Risk Index', gridcolor: 'rgba(255,255,255,0.06)' }
    };

    Plotly.newPlot('chart-route-risks', data, layout, { responsive: true, displayModeBar: false });
}

// -------------------------------------------------------------
// 4. Inventory Trajectory
// -------------------------------------------------------------
function loadInventoryDetail(componentId) {
    const comp = AppState.inventory.find(c => c.component_id === componentId);
    if (!comp) return;

    document.getElementById('inv-current-stock').textContent = `${comp.current_stock} units`;
    document.getElementById('inv-burn-rate').textContent = `Burn rate: ${comp.daily_burn_rate} units/day`;
    document.getElementById('inv-dos').textContent = `${comp.days_of_supply} days`;
    document.getElementById('inv-safety-stock').textContent = `${comp.safety_stock} units`;
    document.getElementById('inv-rop').textContent = `Reorder point: ${comp.reorder_point} units`;
    document.getElementById('inv-stockout-day').textContent = comp.stockout_day ? `Day ${comp.stockout_day}` : "None (Safe)";
    document.getElementById('inv-deficit').textContent = `Cumulative deficit: ${comp.cumulative_deficit_units} units`;

    renderInventoryCurve(comp);
}

function renderInventoryCurve(comp) {
    const days = comp.trajectory.map(t => t.day);
    const stockLevels = comp.trajectory.map(t => t.inventory_level);
    const safetyStocks = comp.trajectory.map(t => t.safety_stock_threshold);
    const rops = comp.trajectory.map(t => t.reorder_point);

    const data = [
        {
            name: `Projected Stock`,
            type: 'scatter',
            mode: 'lines+markers',
            x: days,
            y: stockLevels,
            line: { color: '#0284c7', width: 3 },
            fill: 'tozeroy',
            fillcolor: 'rgba(2, 132, 199, 0.1)'
        },
        {
            name: `Safety Stock (${comp.safety_stock})`,
            type: 'scatter',
            mode: 'lines',
            x: days,
            y: safetyStocks,
            line: { color: '#f59e0b', dash: 'dash', width: 2 }
        },
        {
            name: `Reorder Point (${comp.reorder_point})`,
            type: 'scatter',
            mode: 'lines',
            x: days,
            y: rops,
            line: { color: '#8b5cf6', dash: 'dot', width: 2 }
        }
    ];

    const layout = {
        ...PLOTLY_THEME,
        title: { text: `<b>30-Day Forward Stock Trajectory: ${comp.component_name}</b>`, font: { size: 14, color: '#ffffff' } },
        xaxis: { title: 'Forward Simulation Day', gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Physical On-Hand Inventory (Units)', gridcolor: 'rgba(255,255,255,0.06)' },
        legend: { orientation: 'h', y: 1.1 }
    };

    Plotly.newPlot('chart-inventory-curve', data, layout, { responsive: true, displayModeBar: false });
}

// -------------------------------------------------------------
// 5. Prescriptive Optimization
// -------------------------------------------------------------
async function recomputeInventoryOptimization(serviceLevel) {
    try {
        const res = await fetch('/api/optimize/inventory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_service_level: serviceLevel, order_cost_usd: 150.0 })
        }).then(r => r.json());

        document.getElementById('opt-holding').textContent = `$${Math.round(res.total_optimized_holding_cost_usd).toLocaleString()}`;
        document.getElementById('opt-wc').textContent = `$${Math.round(res.net_working_capital_freed_usd).toLocaleString()}`;

        const tbody = document.querySelector('#table-opt-inv tbody');
        tbody.innerHTML = '';
        res.components.forEach(c => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><b>${c.name}</b></td>
                <td>${c.current_safety_stock}</td>
                <td class="text-emerald"><b>${c.optimized_safety_stock}</b></td>
                <td class="${c.safety_stock_delta > 0 ? 'text-amber' : 'text-emerald'}">${c.safety_stock_delta > 0 ? '+' : ''}${c.safety_stock_delta}</td>
                <td>${c.optimized_eoq}</td>
                <td class="${c.working_capital_impact_usd >= 0 ? 'text-emerald' : 'text-rose'}">$${Math.round(c.working_capital_impact_usd).toLocaleString()}</td>
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.error('Error optimizing inventory:', err);
    }
}

async function recomputeMaintenanceOptimization(hours) {
    try {
        const res = await fetch('/api/optimize/maintenance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ available_technician_hours: hours, cost_per_tech_hour_usd: 85.0 })
        }).then(r => r.json());

        document.getElementById('sched-hours').textContent = `${res.allocated_technician_hours} / ${res.available_technician_hours} hrs`;
        document.getElementById('sched-avoided').textContent = `$${Math.round(res.total_unplanned_losses_avoided_usd).toLocaleString()}`;
        document.getElementById('sched-roi').textContent = `${res.roi_percentage.toFixed(0)}%`;

        const tbody = document.querySelector('#table-opt-sched tbody');
        tbody.innerHTML = '';
        res.scheduled_interventions.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><b>${item.machine_id} (${item.name})</b></td>
                <td>${item.bay}</td>
                <td><span class="badge badge-${item.risk_level.toLowerCase()}">${(item.failure_probability*100).toFixed(0)}%</span></td>
                <td>${item.required_tech_hours} hrs</td>
                <td class="text-emerald">$${Math.round(item.expected_unplanned_loss_usd).toLocaleString()}</td>
                <td class="text-emerald"><b>$${Math.round(item.net_avoided_loss_usd).toLocaleString()}</b></td>
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.error('Error optimizing maintenance:', err);
    }
}

// -------------------------------------------------------------
// 6. Resilience Section
// -------------------------------------------------------------
function renderResilienceSection() {
    const resData = AppState.resilience.component_resilience;
    const hhiData = AppState.resilience.supplier_concentration;

    document.getElementById('hhi-caption').textContent = 
        `Sourcing HHI: ${Math.round(hhiData.hhi_index).toLocaleString()} (${hhiData.concentration_tier}) across $${Math.round(hhiData.total_annual_spend_usd).toLocaleString()} annual spend.`;

    const tbody = document.querySelector('#table-resilience tbody');
    tbody.innerHTML = '';
    resData.resilience_records.forEach(r => {
        const tr = document.createElement('tr');
        const isVuln = r.resilience_gap_days < 0;
        tr.innerHTML = `
            <td><b>${r.name}</b></td>
            <td>${r.supplier}</td>
            <td>${r.time_to_survive_days}d</td>
            <td>${r.time_to_recover_days}d</td>
            <td class="${isVuln ? 'text-rose' : 'text-emerald'}"><b>${r.resilience_gap_days > 0 ? '+' : ''}${r.resilience_gap_days}d</b></td>
            <td><span class="badge ${isVuln ? 'badge-critical' : 'badge-healthy'}">${r.classification.split(' (')[0]}</span></td>
        `;
        tbody.appendChild(tr);
    });

    renderResilienceMatrixChart(resData.resilience_records);
    renderSupplierHhiChart(hhiData);
}

function renderResilienceMatrixChart(records) {
    const maxVal = Math.max(...records.map(r => Math.max(r.time_to_survive_days, r.time_to_recover_days))) + 4;

    const data = [
        {
            name: 'Parity Boundary (TTS = TTR)',
            type: 'scatter',
            mode: 'lines',
            x: [0, maxVal],
            y: [0, maxVal],
            line: { color: '#94a3b8', dash: 'dash', width: 2 }
        },
        {
            name: 'Crisis Zone (TTS < TTR)',
            type: 'scatter',
            mode: 'lines',
            x: [0, maxVal, 0],
            y: [0, maxVal, maxVal],
            fill: 'toself',
            fillcolor: 'rgba(239, 68, 68, 0.12)',
            line: { color: 'rgba(0,0,0,0)' },
            hoverinfo: 'skip'
        },
        {
            name: 'Components',
            type: 'scatter',
            mode: 'markers+text',
            x: records.map(r => r.time_to_survive_days),
            y: records.map(r => r.time_to_recover_days),
            text: records.map(r => r.name),
            textposition: 'top center',
            marker: {
                size: 14,
                color: records.map(r => r.resilience_gap_days < 0 ? '#ef4444' : (r.resilience_gap_days <= 3 ? '#f59e0b' : '#10b981')),
                line: { color: '#ffffff', width: 1.5 }
            }
        }
    ];

    const layout = {
        ...PLOTLY_THEME,
        title: { text: '<b>BCG Resilience Matrix: Time-to-Survive (TTS) vs Time-to-Recover (TTR)</b>', font: { size: 14, color: '#ffffff' } },
        xaxis: { title: 'Time-to-Survive [TTS] (Days before stockout)', gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Time-to-Recover [TTR] (Days to restore supply)', gridcolor: 'rgba(255,255,255,0.06)' }
    };

    Plotly.newPlot('chart-resilience-matrix', data, layout, { responsive: true, displayModeBar: false });
}

function renderSupplierHhiChart(hhiData) {
    const data = [{
        type: 'pie',
        hole: 0.45,
        labels: hhiData.supplier_breakdown.map(s => s.supplier),
        values: hhiData.supplier_breakdown.map(s => s.annual_spend_usd),
        marker: {
            colors: ['#00875a', '#0284c7', '#38bdf8', '#818cf8', '#f59e0b', '#ef4444', '#ec4899']
        },
        textinfo: 'percent+label',
        textposition: 'inside'
    }];

    const layout = {
        ...PLOTLY_THEME,
        showlegend: false
    };

    Plotly.newPlot('chart-supplier-hhi', data, layout, { responsive: true, displayModeBar: false });
}

// -------------------------------------------------------------
// 7. What-If Simulator Scenarios
// -------------------------------------------------------------
async function runScenario1() {
    const compId = document.getElementById('sim1-comp').value;
    const delayDays = parseInt(document.getElementById('sim1-delay').value);

    try {
        const res = await fetch('/api/simulate/shipping_delay', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ component_id: compId, additional_delay_days: delayDays })
        }).then(r => r.json());

        document.getElementById('sim1-arrival').textContent = `Day ${res.simulated_arrival_day}`;
        document.getElementById('sim1-stockout').textContent = res.simulated_stockout_day ? `Day ${res.simulated_stockout_day}` : 'No Stockout';
        document.getElementById('sim1-deficit').textContent = `${res.inventory_deficit_units} units`;

        const actionsUl = document.getElementById('sim1-actions');
        actionsUl.innerHTML = res.mitigation_actions.map(act => `<li>${act}</li>`).join('');

        // Plotly curve
        const days = res.baseline_trajectory.map(t => t.day);
        const data = [
            {
                name: 'Baseline Stock',
                type: 'scatter',
                mode: 'lines',
                x: days,
                y: res.baseline_trajectory.map(t => t.inventory_level),
                line: { color: '#0284c7', width: 2 }
            },
            {
                name: 'Simulated Disruption',
                type: 'scatter',
                mode: 'lines+markers',
                x: days,
                y: res.simulated_trajectory.map(t => t.inventory_level),
                line: { color: '#ef4444', width: 3 }
            }
        ];

        const layout = {
            ...PLOTLY_THEME,
            title: { text: `<b>Inbound Delay Impact (+${delayDays} Days)</b>`, font: { size: 14, color: '#ffffff' } },
            yaxis: { title: 'Units', gridcolor: 'rgba(255,255,255,0.06)' },
            xaxis: { title: 'Day', gridcolor: 'rgba(255,255,255,0.06)' },
            legend: { orientation: 'h', y: 1.1 }
        };

        Plotly.newPlot('chart-sim1-curve', data, layout, { responsive: true, displayModeBar: false });

    } catch (err) {
        console.error('Error running scenario 1:', err);
    }
}

async function runScenario2() {
    const machineId = document.getElementById('sim2-machine').value;
    const days = parseInt(document.getElementById('sim2-days').value);

    try {
        const res = await fetch('/api/simulate/machinery_downtime', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ machine_id: machineId, downtime_days: days })
        }).then(r => r.json());

        document.getElementById('sim2-capacity').textContent = `${res.lost_production_units.toLocaleString()} units`;
        document.getElementById('sim2-cost').textContent = `$${res.estimated_downtime_cost_usd.toLocaleString()}`;

        const optionsUl = document.getElementById('sim2-options');
        optionsUl.innerHTML = res.recovery_options.map(opt => `<li>${opt}</li>`).join('');

        const data = [{
            type: 'bar',
            x: ['Capacity Lost', 'Backlog Accumulated', 'Downtime Hours'],
            y: [res.lost_production_units, res.backlog_units_accumulated, res.total_downtime_hours],
            marker: { color: ['#ef4444', '#ea580c', '#0284c7'] },
            text: [res.lost_production_units, res.backlog_units_accumulated, `${res.total_downtime_hours}h`],
            textposition: 'outside'
        }];

        const layout = {
            ...PLOTLY_THEME,
            title: { text: `<b>Outage Impact: ${machineId} (${days} Days)</b>`, font: { size: 14, color: '#ffffff' } },
            yaxis: { gridcolor: 'rgba(255,255,255,0.06)' }
        };

        Plotly.newPlot('chart-sim2-bar', data, layout, { responsive: true, displayModeBar: false });

    } catch (err) {
        console.error('Error running scenario 2:', err);
    }
}

async function runScenario3() {
    const compId = document.getElementById('sim3-comp').value;
    const cut = parseInt(document.getElementById('sim3-cut').value);
    const delay = parseInt(document.getElementById('sim3-delay').value);

    try {
        const res = await fetch('/api/simulate/supplier_disruption', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ component_id: compId, supply_reduction_pct: cut, extended_delay_days: delay })
        }).then(r => r.json());

        document.getElementById('sim3-stockout').textContent = res.stockout_day ? `Day ${res.stockout_day}` : 'Buffer Holds';
        document.getElementById('sim3-deficit').textContent = `${res.cumulative_deficit_units} units`;

        const data = [{
            type: 'scatter',
            mode: 'lines+markers',
            x: res.trajectory.map(t => t.day),
            y: res.trajectory.map(t => t.inventory_level),
            line: { color: '#ef4444', width: 3 },
            fill: 'tozeroy',
            fillcolor: 'rgba(239, 68, 68, 0.1)'
        }];

        const layout = {
            ...PLOTLY_THEME,
            title: { text: `<b>Supplier Reduction (-${cut}%, +${delay}d delay)</b>`, font: { size: 14, color: '#ffffff' } },
            yaxis: { title: 'Units', gridcolor: 'rgba(255,255,255,0.06)' },
            xaxis: { title: 'Day', gridcolor: 'rgba(255,255,255,0.06)' }
        };

        Plotly.newPlot('chart-sim3-curve', data, layout, { responsive: true, displayModeBar: false });

    } catch (err) {
        console.error('Error running scenario 3:', err);
    }
}

async function runScenario4() {
    const surge = parseInt(document.getElementById('sim4-surge').value);

    try {
        const res = await fetch('/api/simulate/demand_surge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ demand_change_pct: surge })
        }).then(r => r.json());

        document.getElementById('sim4-stockouts').textContent = `${res.components_with_stockout} / ${res.component_details.length}`;

        const tbody = document.querySelector('#table-sim4 tbody');
        tbody.innerHTML = '';
        res.component_details.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><b>${item.name}</b></td>
                <td>${item.new_daily_burn} units/day</td>
                <td class="${item.simulated_stockout_day ? 'text-rose' : 'text-emerald'}">${item.simulated_stockout_day ? `Day ${item.simulated_stockout_day}` : 'Safe'}</td>
                <td class="text-amber">${item.stockout_days_earlier > 0 ? `${item.stockout_days_earlier} days earlier` : 'No change'}</td>
                <td class="text-rose">${item.cumulative_deficit_units} units</td>
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.error('Error running scenario 4:', err);
    }
}

async function runScenario5() {
    const ship = parseInt(document.getElementById('sim5-ship').value);
    const maint = parseInt(document.getElementById('sim5-maint').value);
    const demand = parseInt(document.getElementById('sim5-demand').value);
    const sup = parseInt(document.getElementById('sim5-sup').value);

    try {
        const res = await fetch('/api/simulate/combined', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                additional_shipping_delay: ship,
                machine_downtime_days: maint,
                demand_change_pct: demand,
                supplier_reduction_pct: sup
            })
        }).then(r => r.json());

        const cmp = res.comparison;
        document.getElementById('sim5-stockouts').textContent = `${cmp.simulated_stockout_components} Components`;
        document.getElementById('sim5-deficit').textContent = `${cmp.simulated_inventory_deficit.toLocaleString()} units`;
        document.getElementById('sim5-lost-prod').textContent = `${cmp.lost_production_units.toLocaleString()} units`;
        document.getElementById('sim5-exposure').textContent = `$${Math.round(cmp.total_combined_financial_exposure_usd).toLocaleString()}`;

        const metrics = ["Stockout Components", "Deficit Units (x100)", "Lost Capacity (x100)"];
        const baselineVals = [cmp.baseline_stockout_components, Math.round(cmp.baseline_inventory_deficit/100), 0];
        const simVals = [cmp.simulated_stockout_components, Math.round(cmp.simulated_inventory_deficit/100), Math.round(cmp.lost_production_units/100)];

        const data = [
            { name: 'Baseline Operations', type: 'bar', x: metrics, y: baselineVals, marker: { color: '#0284c7' } },
            { name: 'Simulated Disruption', type: 'bar', x: metrics, y: simVals, marker: { color: '#ef4444' } }
        ];

        const layout = {
            ...PLOTLY_THEME,
            barmode: 'group',
            title: { text: '<b>Baseline vs Combined Multi-Vector Shock Impact</b>', font: { size: 14, color: '#ffffff' } },
            yaxis: { gridcolor: 'rgba(255,255,255,0.06)' },
            legend: { orientation: 'h', y: 1.1 }
        };

        Plotly.newPlot('chart-sim5-comparison', data, layout, { responsive: true, displayModeBar: false });

    } catch (err) {
        console.error('Error running scenario 5:', err);
    }
}

// -------------------------------------------------------------
// 8. Boardroom Executive Memo
// -------------------------------------------------------------
function renderExecutiveMemo() {
    if (!AppState.memoData) return;

    document.getElementById('memo-content').textContent = AppState.memoData.memo_markdown;

    const playbook = document.getElementById('playbook-list');
    playbook.innerHTML = '';

    AppState.memoData.recommendations.forEach(rec => {
        const div = document.createElement('div');
        div.className = 'card-section mb-3';
        const priorityColor = rec.priority === 'CRITICAL' ? '#ef4444' : (rec.priority === 'HIGH' ? '#ea580c' : '#f59e0b');

        div.innerHTML = `
            <div class="flex-between mb-2">
                <span style="font-size:15px; font-weight:700; color:#38bdf8;">[${rec.category}] ${rec.target}</span>
                <span class="badge" style="background:${priorityColor}22; border:1px solid ${priorityColor}; color:${priorityColor};">${rec.priority} PRIORITY</span>
            </div>
            <p><b>Finding:</b> ${rec.finding}</p>
            <p class="mt-2 text-slate-200"><b>Action:</b> ${rec.action}</p>
            <p class="mt-1 text-emerald"><b>Impact:</b> ${rec.estimated_impact}</p>
        `;
        playbook.appendChild(div);
    });
}
