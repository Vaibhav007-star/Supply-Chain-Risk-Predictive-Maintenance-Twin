# MASTER PROMPT: SUPPLY CHAIN RISK & PREDICTIVE MAINTENANCE TWIN

## 1. YOUR ROLE

Act as a senior AI/ML engineer, data scientist, supply chain analyst, predictive maintenance specialist, and Python dashboard developer.

Build a practical, data-driven project named:

**Supply Chain Risk & Predictive Maintenance Twin**

Develop the project inside my existing VS Code environment using Antigravity. Work step by step, inspect my existing files and installed software before making changes, and explain important decisions in simple language.

Do not immediately generate the entire project without inspecting the environment and planning the implementation.

## 2. PROJECT OBJECTIVE

Build an AI-powered decision-support system for manufacturing companies that helps managers anticipate:

1. Potential machinery failures before equipment breaks down.
2. Supply chain delays before they seriously affect production.
3. Inventory shortages caused by delayed shipments or unexpected demand.
4. Weather-related transportation disruptions.
5. Operational risks caused by supplier dependency and geopolitical disruptions.

The system must combine predictive analytics, risk scoring, inventory analysis, and an interactive digital-twin-style simulation dashboard.

The main objective is to help manufacturing managers make informed decisions before disruptions occur.

Important: Predictions must be presented as estimates with appropriate uncertainty, not as guaranteed future events.

## 3. FIRST: INSPECT MY ENVIRONMENT

Before coding:

- Inspect the current VS Code workspace and existing project files.
- Identify the installed Python version and available packages.
- Check whether a dataset has already been downloaded.
- Reuse relevant existing files instead of unnecessarily creating duplicates.
- Identify whether the project is empty or partially implemented.
- Provide a concise implementation plan and wait for my approval before making major structural changes.

Do not delete existing files or overwrite datasets without permission.

If a required dataset is missing, recommend an appropriate public dataset and explain its purpose before proceeding.

## 4. DATA SOURCES AND DATA MANAGEMENT

The project should support both real datasets and a clearly labeled synthetic dataset for initial testing.

Consider these data categories:

### A. Machinery telemetry
- Machine ID
- Timestamp
- Temperature
- Vibration
- Rotational speed
- Operating hours
- Pressure, where available
- Maintenance history
- Failure or fault labels, where available

### B. Supply chain and logistics
- Shipment ID
- Supplier ID
- Product or component ID
- Shipment origin and destination
- Planned and actual delivery dates
- Shipping duration
- Quantity shipped
- Supplier lead time
- Delay status
- Inventory levels
- Daily demand
- Reorder thresholds

### C. Weather and external disruption data
- Temperature and severe weather conditions
- Rainfall or storm indicators, where available
- Transport route disruption indicators
- Supplier-region risk
- Geopolitical disruption indicators, where a reliable source is available

Use public data sources or documented APIs when practical. Do not assume that an API is available or that live data can be accessed without credentials.

Keep external data separate from operational datasets and merge records using appropriate timestamps, geographic identifiers, and entity IDs.

Do not invent real-world disruption data. If synthetic data is used, clearly label it as simulated.

## 5. MACHINE LEARNING AND PREDICTIVE ANALYTICS

Implement the simplest reliable model supported by the available data. Compare more advanced models only when they provide meaningful value.

### A. Predictive maintenance

Build a model to estimate machine failure risk or remaining useful life, depending on what the dataset supports.

Potential approaches:
- Random Forest or XGBoost for failure classification.
- ARIMA or Prophet for suitable time-series forecasting.
- LSTM only when sufficient sequential data and computing resources are available.

Requirements:
- Identify the prediction target before training.
- Use chronological train/validation/test splits for time-dependent data.
- Avoid data leakage from future observations.
- Evaluate classification with precision, recall, F1-score, and ROC-AUC where appropriate.
- Evaluate forecasts with MAE, RMSE, or other suitable metrics.
- Display prediction probabilities only when the model supports calibrated probability estimates.
- Explain the important features associated with elevated risk.

Do not claim that a machine will fail at an exact time unless the data and model genuinely support that estimate.

### B. Supply chain delay prediction

Build a model to estimate shipment delays or expected delivery time.

Consider:
- Historical supplier lead times
- Shipping route
- Shipment characteristics
- Seasonal patterns
- Weather disruptions
- Supplier reliability
- Historical delay frequency

Use an appropriate classification or regression model based on the target variable.

Evaluate performance on unseen future-period data.

Display predicted delay risk, estimated delay duration where supported, and the factors contributing to the prediction.

### C. Inventory risk forecasting

Estimate future inventory using:
- Current inventory
- Historical demand
- Forecast demand
- Supplier lead time
- Outstanding shipments
- Safety stock
- Reorder point

Identify possible stockout dates or periods when inventory may fall below the required threshold.

Use actual historical data when available. If data is simulated, display that fact prominently.

## 6. SUPPLY CHAIN DIGITAL TWIN AND WHAT-IF SIMULATOR

Create a digital-twin-style simulation of the available manufacturing and supply chain data.

The simulator must allow users to change scenario parameters and observe the resulting calculations immediately.

Include the following scenarios:

### Scenario 1: Shipping delay
Question: What happens if a shipment is delayed by five days?

Allow users to modify the delay from 0 to 30 days.

Calculate and display:
- Revised estimated arrival date
- Projected inventory levels
- Potential stockout date
- Inventory deficit, if any
- Estimated production impact, when the required data is available
- Suggested mitigation actions

### Scenario 2: Machinery failure
Question: What happens if a critical machine becomes unavailable?

Allow users to simulate downtime duration.

Estimate:
- Production capacity lost
- Potential production backlog
- Component consumption changes, if modeled
- Estimated operational cost, if cost inputs are available
- Potential maintenance or recovery options

### Scenario 3: Supplier disruption
Question: What happens if a primary supplier cannot deliver?

Allow users to simulate a supplier delay or reduced supply quantity.

Show:
- Inventory exposure
- Affected components
- Potential shortages
- Alternative supplier or inventory strategies, when relevant data exists

### Scenario 4: Demand increase
Question: What happens if demand increases unexpectedly?

Allow users to change demand by a configurable percentage.

Show:
- Forecast inventory depletion
- Stockout risk
- Additional replenishment requirements
- Potential changes to safety stock

### Scenario 5: Combined disruption
Allow users to combine shipping delays, machine downtime, and increased demand.

Compare the combined scenario with the baseline scenario.

IMPORTANT:
Use transparent calculations and documented assumptions. A simulated outcome is not an observed real-world outcome. Do not represent scenario results as verified predictions.

## 7. INTERACTIVE DASHBOARD

Build an attractive, professional, responsive dashboard using Python Plotly.

Use Streamlit for the application interface unless the existing workspace already uses a suitable framework.

Organize the dashboard into the following sections:

### A. Executive overview
Display:
- Total machines monitored
- High-risk machines
- At-risk shipments
- Components with potential stockouts
- Predicted operational disruptions
- Estimated exposure to operational losses, only when sufficient cost data exists

### B. Predictive maintenance
Display:
- Machine health and risk overview
- Machine-level risk rankings
- Telemetry trends
- Predicted risk over time, where supported
- Actual versus predicted outcomes
- Model evaluation metrics

### C. Supply chain monitoring
Display:
- Shipment status
- Predicted delivery delays
- Supplier performance
- Lead-time trends
- Route and disruption risk
- Inventory availability by component

### D. Inventory forecasting
Display:
- Historical and forecast demand
- Projected inventory
- Reorder thresholds
- Safety stock
- Estimated stockout windows
- Outstanding inbound shipments

### E. What-if simulation
Include interactive controls for:
- Shipping delay duration
- Machine downtime duration
- Demand change percentage
- Supplier delivery reduction
- Safety stock adjustments

Use Plotly charts to compare baseline results with simulated outcomes.

The charts should update when users change the scenario inputs.

### F. Recommendations
Provide explainable, data-supported recommendations, such as:
- Inspect a high-risk machine.
- Expedite a critical shipment.
- Increase safety stock for a vulnerable component.
- Evaluate an alternative supplier.
- Reschedule production to reduce disruption.

Recommendations must be based on actual model outputs or transparent rules, not fabricated AI reasoning.

## 8. TECHNICAL ARCHITECTURE

Use a clean, modular Python architecture.

Suggested structure:

project_root/
    app.py
    requirements.txt
    README.md
    config/
    data/
        raw/
        processed/
    models/
    src/
        data_loader.py
        data_validation.py
        preprocessing.py
        maintenance_model.py
        delay_prediction.py
        inventory_forecasting.py
        scenario_simulator.py
        risk_scoring.py
        visualization.py
    tests/

Adapt this structure to the existing workspace rather than creating unnecessary folders.

Recommended libraries:
- pandas
- numpy
- scikit-learn
- plotly
- streamlit
- statsmodels
- joblib
- pytest

Install additional packages only when required by the selected approach.

Use Prophet, XGBoost, or LSTM only when their benefits justify the additional dependencies and the data supports them.

Keep data processing, model training, simulation calculations, and dashboard presentation in separate modules.

## 9. MODEL RELIABILITY AND DATA QUALITY

Implement:
- Missing-value handling
- Duplicate detection
- Timestamp validation
- Outlier investigation
- Consistent units
- Data schema validation
- Reproducible model training
- Chronological validation where appropriate
- Clear baseline comparisons

If a dataset lacks machine failure labels, do not pretend that supervised failure prediction has been trained successfully.

Instead, implement an appropriate anomaly-detection or rule-based monitoring approach and clearly explain its limitations.

If historical shipment data is unavailable, use a documented baseline or simulation until suitable data is obtained.

Prevent future information from leaking into model training.

## 10. TESTING AND VERIFICATION

Test the project incrementally.

Verify that:
- The application starts successfully.
- Dataset loading and validation work.
- Models train using the selected datasets.
- Evaluation metrics are calculated correctly.
- Forecasts use the intended time horizon.
- Scenario calculations are internally consistent.
- Changing scenario parameters updates the outputs.
- Charts handle missing data without crashing.
- The application remains usable when optional external data is unavailable.

Create tests for the core inventory, delay, and downtime calculations.

Run the tests and fix errors before describing a feature as complete.

Do not claim that a test passed unless it was actually executed.

## 11. USER INTERFACE AND DESIGN

Use a modern industrial analytics design.

Design preferences:
- Clean layout with clear section headings
- Consistent typography and spacing
- Professional charts and KPI cards
- Readable labels, legends, and tooltips
- Appropriate warning indicators for elevated risks
- Filters for machine, supplier, component, and date
- Helpful explanations for business users
- Clear distinction between observed, predicted, and simulated values

Avoid clutter, excessive animations, and unnecessary decorative elements.

Ensure the dashboard is understandable to a manufacturing manager who does not have a machine learning background.

## 12. SECURITY AND PERFORMANCE

- Never hardcode API keys or credentials.
- Use environment variables for optional external services.
- Do not upload private company telemetry or supplier data to third-party services without explicit authorization.
- Cache expensive data loading or model operations when appropriate.
- Avoid retraining models every time a user changes a simulation slider.
- Use saved model artifacts when suitable.
- Handle missing files, invalid inputs, and API failures gracefully.

## 13. IMPLEMENTATION WORKFLOW

Follow these phases in order.

Phase 1: Inspect the workspace, Python environment, and available datasets.

Phase 2: Recommend suitable datasets, identify prediction targets, and define the initial scope.

Phase 3: Prepare and validate the data.

Phase 4: Build a simple baseline for machine risk, shipment delays, and inventory forecasting according to data availability.

Phase 5: Evaluate the models and improve them only where justified.

Phase 6: Implement transparent scenario calculations.

Phase 7: Build the interactive Plotly and Streamlit dashboard.

Phase 8: Integrate the modules and run tests.

Phase 9: Improve the interface, document assumptions, and verify the final workflow.

Phase 10: Prepare the README with installation instructions, dataset descriptions, model limitations, and startup commands.

At the end of each phase:
- Summarize what was completed.
- Identify the files created or changed.
- Report actual test results.
- Explain any limitations.
- State the next recommended step.

Do not silently skip a phase or claim that an unfinished feature is complete.

## 14. FINAL ACCEPTANCE CRITERIA

The project is considered complete only when:

1. It runs locally in my VS Code environment.
2. The data sources and their limitations are documented.
3. At least one valid predictive or risk-monitoring approach is implemented for each supported domain.
4. The models are evaluated using appropriate metrics when ground-truth data exists.
5. The inventory calculations are logically consistent.
6. Users can run shipping-delay and downtime simulations.
7. Baseline and simulated outcomes can be compared.
8. The dashboard presents understandable, interactive Plotly visualizations.
9. Important assumptions and uncertainty are visible.
10. The application has a usable README and has passed its implemented tests.

If a requirement cannot be satisfied because of missing data, missing credentials, or environmental limitations, document the blocker instead of fabricating results.

## 15. FIRST RESPONSE REQUIRED

For your first response, do not write the complete codebase.

Inspect my existing workspace and provide:

1. A short summary of the current environment.
2. The datasets already available, if any.
3. The recommended initial model and why.
4. Any missing dependencies or data requirements.
5. A phased implementation plan.

Keep the explanation simple, practical, and focused on this project.

Wait for my approval before beginning major implementation changes.