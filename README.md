# Supply Chain Risk & Predictive Maintenance Twin (BCG X Edition)

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://supply-chain-risk-predictive-mainte.vercel.app)
[![API Status](https://img.shields.io/badge/API-Active%20%26%20Healthy-059669?style=for-the-badge&logo=fastapi&logoColor=white)](https://supply-chain-risk-predictive-mainte.vercel.app/api/health)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

> 🌐 **Live Web Application**: [https://supply-chain-risk-predictive-mainte.vercel.app](https://supply-chain-risk-predictive-mainte.vercel.app)

An enterprise-grade AI decision-support twin and executive prescriptive command cockpit built for advanced manufacturing and global supply chains.

The platform bridges shopfloor machine telemetry, inbound shipping delay risks, and dynamic inventory trajectories with **BCG X-standard Value-at-Risk (VaR) modeling, Explainable AI (XAI) root-cause attribution, prescriptive policy optimization, and network resilience frameworks (TTS vs TTR)**.

---

## 🏭 System Architecture

```
Supply Chain Risk & Predictive Maintenance Twin/
│
├── app.py                      # BCG X Executive Decision Cockpit (Streamlit & Plotly)
├── requirements.txt            # Production dependencies
├── config/
│   ├── __init__.py
│   └── settings.py             # Global constants, paths, thresholds, and seeds
│
├── data/
│   ├── raw/                    # Raw source CSV datasets
│   │   ├── ai4i2020.csv
│   │   └── DataCoSupplyChainDataset.csv
│   └── processed/              # High-speed validated Parquet datasets & Twin topology
│       ├── maintenance_processed.parquet
│       ├── supply_chain_processed.parquet
│       └── twin_operational_state.json
│
├── models/                     # Serialized machine learning bundles (joblib)
│   ├── maintenance_classifier.joblib
│   └── delay_risk_classifier.joblib
│
├── src/                        # Modular analytics, optimization, and XAI pipelines
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion and twin operational state generator
│   ├── data_validation.py      # Schema, bounds, and boundary checks
│   ├── preprocessing.py        # Physics-based telemetry & supply chain feature engineering
│   ├── maintenance_model.py    # Random Forest failure classifier & physics diagnostics
│   ├── delay_prediction.py     # HistGradientBoosting delay classifier & duration regressor
│   ├── inventory_forecasting.py# Deterministic day-by-day 30-day stockout projection engine
│   ├── scenario_simulator.py   # What-If simulator (5 distinct disruption scenarios)
│   ├── risk_scoring.py         # Multi-factor plant health scoring & recommendation engine
│   ├── prescriptive_optimization.py # BCG X: Safety Stock / EOQ & Maintenance Scheduling Optimizer
│   ├── explainability.py       # BCG X: Local feature attribution & XAI root-cause waterfalls
│   ├── resilience_metrics.py   # BCG X: Time-to-Survive (TTS), Time-to-Recover (TTR), & Supplier HHI
│   └── visualization.py        # Plotly dark-themed executive charts & waterfall graphs
│
└── tests/
    └── test_pipeline.py        # Automated test suite (Pytest - 13 tests)
```

---

## 💎 BCG X Core Strategic Advancements

### 1. Financial Value-at-Risk (VaR) & ROI Arbitrage
Moves beyond raw predictive scores to quantify executive business value and EBITDA impact:
- **Reactive Status Quo Exposure**: Quantifies catastrophic downtime losses ($650 - $1,500/hr + 16-24 hr halts), late order stockout penalties ($220/unit), and emergency spot freight.
- **Prescriptive AI Interventions**: Low-cost planned shift-change maintenance, buffer right-sizing, and proactive carrier rerouting.
- **Boardroom Value Metrics**: Direct computation of **Annualized EBITDA Impact**, **ROI Multiplier (3.5x - 5.0x)**, and **Payback Horizon in Weeks**.

### 2. Mathematical Prescriptive Decision Optimizer (`src/prescriptive_optimization.py`)
- **Stochastic Safety Stock & EOQ Engine**: Optimizes buffer sizes and Economic Order Quantities by solving the trade-off between stockout risk and working capital carrying cost under lead-time and demand variance across target service levels (90%, 95%, 98%, 99%).
- **Constrained Maintenance Intervention Scheduler**: Solves the resource-constrained knapsack problem across plant equipment, prioritizing assets by failure progression and hourly downtime loss to maximize avoided loss within limited technician crew hours.

### 3. Explainable AI (XAI) & Root Cause Attribution (`src/explainability.py`)
Rooted in BCG X's AI heritage, every prediction is paired with transparent, local attribution:
- **Physics-Grounded Telemetry Attribution**: Decomposes machine failure probability into constituent physical deviations: Spindle Speed delta, Mechanical Power anomalies, Tool Wear degradation, Thermal Heat Dissipation ($\Delta T$), and Overstrain factor.
- **Supply Chain Lead Time Attribution**: Decomposes shipment late delivery risk into Carrier baseline, Weather/Climate disruption index, and Geopolitical friction.
- **Interactive Waterfall Visualizations**: Clear Plotly waterfall charts for shopfloor engineers and plant leadership.

### 4. BCG Supply Chain Resilience Framework (`src/resilience_metrics.py`)
- **Time-to-Survive (TTS)**: The number of operating days before complete inventory starvation halts production if a node is severed.
- **Time-to-Recover (TTR)**: The calendar days required to qualify, procure, and restore supply via alternative sources or emergency repairs.
- **Resilience Gap ($\Delta = \text{TTS} - \text{TTR}$)**: Identifies structural crisis nodes where $\text{TTS} < \text{TTR}$.
- **Supplier Concentration (HHI)**: Computes the Herfindahl-Hirschman Index across annual procurement spend to detect single-point-of-failure vulnerabilities.

### 5. Automated Boardroom Executive Briefing Memo
Generates a downloadable, presentation-ready executive briefing memo in BCG consulting standard, detailing:
- Executive Summary & Quantified Value Capture
- Resilience Matrix Profile (TTS vs TTR & Supplier HHI)
- Tactical Day 0-7 Immediate Mitigations
- Strategic 30-90 Day Transformation Roadmap

---

## 📊 Real-World Datasets

1. **AI4I 2020 Predictive Maintenance Dataset** (`stephanmatzka/predictive-maintenance-dataset-ai4i-2020`):
   - **Records**: 10,000 real equipment observations.
   - **Features**: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Product quality variant (L/M/H).
   - **Engineered Physics Metrics**: Temperature differential ($\Delta T$), Mechanical Power ($kW$), Overstrain Factor ($\text{Torque} \times \text{Tool wear}$).
   - **Target**: `machine_failure` with failure mode breakdown (*TWF, HDF, PWF, OSF, RNF*).

2. **DataCo Global Supply Chain for Big Data Analysis** (`shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`):
   - **Records**: 180,519 supply chain shipments.
   - **Features**: Real vs scheduled shipping days, shipping modes, customer order regions, order quantities, sales volume.
   - **Target**: `Late_delivery_risk` and continuous delay days duration.

---

## 🤖 Machine Learning Models & Validated Metrics

### 1. Machinery Failure Classifier (`maintenance_classifier.joblib`)
- **Algorithm**: Balanced Random Forest with physical feature engineering.
- **Accuracy**: **98.55%**
- **ROC-AUC**: **0.9806**
- **Failure Recall**: **71.79%** (despite heavy class imbalance: 3.39% failure baseline).
- **Key Risk Drivers**: Rotational speed ($17.4\%$), Mechanical power ($16.4\%$), Torque ($14.7\%$), Tool wear ($12.0\%$), Overstrain factor ($8.6\%$).
- **Rule-Based Failure Diagnostics**:
  - *Tool Wear Failure (TWF)*: Tool wear $\ge 200$ min.
  - *Heat Dissipation Failure (HDF)*: Process $-$ Air Temp $< 8.6$ K and speed $< 1380$ rpm.
  - *Power Failure (PWF)*: Mechanical power $< 3.5$ kW or $> 9.0$ kW.
  - *Overstrain Failure (OSF)*: Tool wear $\times$ Torque $> 11,000$.

### 2. Supply Chain Delay Model (`delay_risk_classifier.joblib`)
- **Algorithm**: Histogram Gradient Boosted Classifier & Regressor.
- **Precision**: **83.69%**
- **ROC-AUC**: **0.7363**
- **Delay Duration MAE**: **0.98 days** (predicts shipment delay duration within ~1 day).

---

## 🔮 What-If Digital Twin Simulator Scenarios

The simulator provides deterministic, transparent calculations for 5 core scenarios:
1. **Scenario 1: Shipping Delay (0 to 30 days)**: Revised delivery day, dynamic day-by-day inventory curve, exact stockout date, and cumulative deficit.
2. **Scenario 2: Machinery Downtime (1 to 14 days)**: Lost production capacity, accumulated backlog, downtime losses ($USD), and recovery routing.
3. **Scenario 3: Supplier Disruption (0 to 100% reduction, prolonged shock)**: Supplier quota reductions, inventory exposure, and dual-sourcing reallocation.
4. **Scenario 4: Unexpected Demand Surge (-50% to +100%)**: Accelerated depletion, safety stock adjustment recommendations, and stockout vulnerability.
5. **Scenario 5: Combined Multi-Vector Stress Test**: Multi-vector simultaneous disruption with side-by-side grouped baseline comparisons.

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12.10)
- Windows / macOS / Linux

### 2. Activate Virtual Environment & Install Dependencies
```bash
# If using the existing venv:
.\.venv\Scripts\activate

# Or install from requirements:
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest -v tests/
```
All **24 unit and integration tests** validate dataset boundaries, model inferences, inventory calculations, scenario simulations, prescriptive optimization, explainability attribution, resilience metrics, and FastAPI REST endpoints.

### 4. Launch the Interactive Web Application
```bash
python server.py
```
Open your browser at `http://127.0.0.1:8000`.

---

## ⚠️ Decision-Support & Modeling Limitations

- **Probabilistic Nature**: All machine failure risks and shipment delay estimates are probabilistic approximations with uncertainty bands. They are intended for proactive managerial risk mitigation and should not be treated as guaranteed deterministic events.
- **Operational Synthesis Layer**: Factory machine topologies (Bay A, Bay B, Bay C) are mapped to real sensor records from AI4I 2020 and DataCo to simulate realistic multi-echelon interactions without exposing proprietary industrial intellectual property.
