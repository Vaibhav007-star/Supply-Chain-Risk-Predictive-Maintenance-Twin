"""
Feature engineering and preprocessing pipelines for predictive maintenance and supply chain data.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def preprocess_maintenance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and feature-engineers telemetry for predictive maintenance.
    Calculates physical parameters (Delta-T, Mechanical Power, Overstrain factor).
    """
    df = df.copy()

    # Rename columns to clean snake_case
    rename_map = {
        'UDI': 'udi',
        'Product ID': 'product_id',
        'Type': 'product_type',
        'Air temperature [K]': 'air_temperature_k',
        'Process temperature [K]': 'process_temperature_k',
        'Rotational speed [rpm]': 'rotational_speed_rpm',
        'Torque [Nm]': 'torque_nm',
        'Tool wear [min]': 'tool_wear_min',
        'Machine failure': 'machine_failure',
        'TWF': 'fail_tool_wear',
        'HDF': 'fail_heat_dissipation',
        'PWF': 'fail_power',
        'OSF': 'fail_overstrain',
        'RNF': 'fail_random'
    }
    df = df.rename(columns=rename_map)

    # 1. Temperature difference (Critical for Heat Dissipation Failure HDF)
    df['temp_difference_k'] = df['process_temperature_k'] - df['air_temperature_k']

    # 2. Mechanical Power (W) = Torque (Nm) * Angular velocity (rad/s)
    # rad/s = rpm * (2 * pi / 60) ~= rpm * 0.104719755
    df['power_kw'] = (df['torque_nm'] * df['rotational_speed_rpm'] * (2 * np.pi / 60.0)) / 1000.0

    # 3. Overstrain Factor (Tool wear * Torque) - Critical for Overstrain Failure OSF
    df['overstrain_factor'] = df['tool_wear_min'] * df['torque_nm']

    # 4. Temperature ratio
    df['temp_ratio'] = df['process_temperature_k'] / (df['air_temperature_k'] + 1e-6)

    # 5. Torque to speed ratio
    df['torque_speed_ratio'] = df['torque_nm'] / (df['rotational_speed_rpm'] + 1e-6)

    # One-hot encode product_type (L=Low quality variant, M=Medium, H=High)
    df['type_H'] = (df['product_type'] == 'H').astype(int)
    df['type_M'] = (df['product_type'] == 'M').astype(int)
    df['type_L'] = (df['product_type'] == 'L').astype(int)

    return df


def preprocess_supply_chain_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and feature-engineers supply chain transaction records.
    Calculates actual shipping delay, lead times, order dates, and route features.
    """
    df = df.copy()

    # Standardize column names
    col_map = {
        'Days for shipping (real)': 'shipping_days_real',
        'Days for shipment (scheduled)': 'shipping_days_scheduled',
        'Benefit per order': 'benefit_per_order',
        'Delivery Status': 'delivery_status',
        'Late_delivery_risk': 'late_delivery_risk',
        'Category Name': 'category_name',
        'Customer City': 'customer_city',
        'Customer Country': 'customer_country',
        'Customer Segment': 'customer_segment',
        'Department Name': 'department_name',
        'Order City': 'order_city',
        'Order Country': 'order_country',
        'Order Region': 'order_region',
        'Order State': 'order_state',
        'Order Status': 'order_status',
        'Product Name': 'product_name',
        'Product Price': 'product_price',
        'Shipping Mode': 'shipping_mode',
        'order date (DateOrders)': 'order_date',
        'shipping date (DateOrders)': 'shipping_date',
        'Order Item Quantity': 'order_item_quantity',
        'Order Item Total': 'order_item_total',
        'Sales': 'sales'
    }

    cols_to_keep = [c for c in col_map.keys() if c in df.columns]
    df = df[cols_to_keep].rename(columns={c: col_map[c] for c in cols_to_keep})

    # Parse dates safely
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['shipping_date'] = pd.to_datetime(df['shipping_date'], errors='coerce')

    # Drop records with invalid order dates
    df = df.dropna(subset=['order_date'])

    # Delay in days
    df['delay_days'] = df['shipping_days_real'] - df['shipping_days_scheduled']
    df['is_delayed'] = (df['delay_days'] > 0).astype(int)

    # Temporal features
    df['order_year'] = df['order_date'].dt.year
    df['order_month'] = df['order_date'].dt.month
    df['order_dayofweek'] = df['order_date'].dt.dayofweek
    df['order_quarter'] = df['order_date'].dt.quarter

    # Lead time variance ratio
    df['shipping_days_ratio'] = df['shipping_days_real'] / (df['shipping_days_scheduled'] + 1e-4)

    return df
