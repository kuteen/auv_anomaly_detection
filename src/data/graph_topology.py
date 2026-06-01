"""
Enhanced Graph Topology and Feature Engineering for AUV Sensor Networks

Based on Slocum G2 glider domain knowledge:
- Navigation physics: lat/lon change based on heading + speed
- Vehicle dynamics: pitch → depth rate, roll → heading efficiency  
- Energy: battery state affects all operations
- Science: water properties are spatially correlated

Author: Enhanced for AUV Anomaly Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json


# ============================================================
# EXPLICIT GRAPH TOPOLOGY - Domain Knowledge Based
# ============================================================

# Sensor index mapping (must match the sensor list in config)
SENSOR_TO_IDX = {
    'm_altitude': 0,
    'm_ballast_pumped': 1,
    'm_battery': 2,
    'm_battery_inst': 3,
    'm_battpos': 4,
    'm_coulomb_amphr_total': 5,
    'm_depth': 6,
    'm_final_water_vx': 7,
    'm_final_water_vy': 8,
    'm_heading': 9,
    'm_lat': 10,
    'm_lon': 11,
    'm_leakdetect_voltage': 12,
    'm_pitch': 13,
    'm_roll': 14,
    'm_speed': 15,
    'sci_water_cond': 16,
    'sci_water_pressure': 17,
    'sci_water_temp': 18
}

IDX_TO_SENSOR = {v: k for k, v in SENSOR_TO_IDX.items()}

# ============================================================
# DOMAIN-KNOWLEDGE GRAPH EDGES
# ============================================================

# Define explicit edges based on physical relationships in underwater gliders
# Format: (source_idx, target_idx, weight)

GRAPH_EDGES = {
    # === NAVIGATION GROUP ===
    # Position changes based on heading and speed
    ('m_heading', 'm_lat'): 1.0,      # Heading affects latitude trajectory
    ('m_heading', 'm_lon'): 1.0,      # Heading affects longitude trajectory  
    ('m_speed', 'm_lat'): 0.8,         # Speed affects position change rate
    ('m_speed', 'm_lon'): 0.8,
    
    # Ground truth vs estimated
    ('m_lat', 'm_final_water_vx'): 0.5,  # Current estimation
    ('m_lon', 'm_final_water_vy'): 0.5,
    
    # === VEHICLE DYNAMICS GROUP ===
    # Pitch controls depth rate
    ('m_pitch', 'm_depth'): 1.0,      # Pitch directly affects depth
    ('m_depth', 'm_pitch'): 0.3,      # Depth feedback
    
    # Roll affects heading efficiency
    ('m_roll', 'm_heading'): 0.6,     # Roll affects heading accuracy
    
    # Ballast controls buoyancy → depth
    ('m_ballast_pumped', 'm_depth'): 0.9,
    ('m_ballast_pumped', 'm_pitch'): 0.4,
    
    # === ENERGY GROUP ===
    # All systems affected by battery state
    ('m_battery', 'm_battery_inst'): 1.0,
    ('m_battery', 'm_speed'): 0.5,     # Battery affects available power
    ('m_battery', 'm_ballast_pumped'): 0.3,
    
    # Coulomb counting tracks energy use
    ('m_coulomb_amphr_total', 'm_battery'): 0.7,
    ('m_coulomb_amphr_total', 'm_speed'): 0.4,
    
    # Battery position affects vehicle balance
    ('m_battpos', 'm_pitch'): 0.5,
    ('m_battpos', 'm_roll'): 0.5,
    
    # === SCIENCE SENSORS GROUP ===
    # Water properties are spatially correlated (same water mass)
    ('sci_water_temp', 'sci_water_pressure'): 1.0,  # Thermocline
    ('sci_water_temp', 'sci_water_cond'): 1.0,       # T-S relationship
    ('sci_water_pressure', 'sci_water_cond'): 0.9,   # Density relationship
    
    # Depth affects all science readings
    ('m_depth', 'sci_water_temp'): 0.7,
    ('m_depth', 'sci_water_pressure'): 1.0,
    ('m_depth', 'sci_water_cond'): 0.6,
    
    # === ALTITUDE ===
    # Altitude inversely related to depth (when diving, altitude = 0)
    ('m_depth', 'm_altitude'): 0.8,
    
    # === LEAK DETECTION ===
    # Leak affects multiple systems
    ('m_leakdetect_voltage', 'm_battery'): 0.4,
    ('m_leakdetect_voltage', 'm_ballast_pumped'): 0.2,
}


def build_adjacency_matrix(n_sensors: int = 19) -> np.ndarray:
    """
    Build adjacency matrix from domain knowledge edges.
    
    Returns:
        A: Adjacency matrix [n_sensors, n_sensors]
    """
    A = np.zeros((n_sensors, n_sensors), dtype=np.float32)

    # Write each domain edge symmetrically so the graph stays undirected,
    # skipping any sensor name that is not in the current index mapping.
    for (src, tgt), weight in GRAPH_EDGES.items():
        src_idx = SENSOR_TO_IDX.get(src)
        tgt_idx = SENSOR_TO_IDX.get(tgt)
        if src_idx is not None and tgt_idx is not None:
            A[src_idx, tgt_idx] = weight
            A[tgt_idx, src_idx] = weight  # Undirected graph
    
    # NOTE: Do NOT add self-loops here.
    # normalise_adjacency() in graph_builders.py adds self-loops via A_hat = A + I.
    # Adding them here as well would create diagonal values of 2.0, biasing
    # the GCN toward self-features and suppressing inter-sensor message passing.
    
    return A


def get_edge_list() -> List[Tuple[int, int, float]]:
    """Get edge list format."""
    edges = []
    for (src, tgt), weight in GRAPH_EDGES.items():
        src_idx = SENSOR_TO_IDX.get(src)
        tgt_idx = SENSOR_TO_IDX.get(tgt)
        if src_idx is not None and tgt_idx is not None:
            edges.append((src_idx, tgt_idx, weight))
    return edges


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

def compute_temporal_features(df: pd.DataFrame, 
                            time_col: str = 'time',
                            dt: float = 5.0) -> pd.DataFrame:
    """
    Compute temporal/delta features for each sensor.
    
    Features added:
    - delta_{col}: First-order difference (rate of change)
    - rolling_mean_{col}: Rolling average over window
    - rolling_std_{col}: Rolling standard deviation
    
    Args:
        df: Input DataFrame
        time_col: Name of time column
        dt: Time step in seconds
        
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    sensor_cols = [c for c in df.columns if c != time_col and c not in 
                   ['year', 'julian_day', 'dive_cycle']]
    
    # Delta features (rate of change)
    for col in sensor_cols:
        if col in df.columns:
            df[f'delta_{col}'] = df[col].diff() / dt
            df[f'delta_{col}'] = df[f'delta_{col}'].fillna(0)
    
    # Rolling features (window = 10 timesteps = 50 seconds)
    window = 10
    for col in sensor_cols:
        if col in df.columns:
            df[f'rolling_mean_{col}'] = df[col].rolling(window, min_periods=1).mean()
            df[f'rolling_std_{col}'] = df[col].rolling(window, min_periods=1).std().fillna(0)
    
    return df


def compute_derived_features(df: pd.DataFrame,
                           time_col: str = 'time') -> pd.DataFrame:
    """
    Compute physics-based derived features for underwater gliders.
    
    Derived features:
    - distance_traveled: Cumulative distance from lat/lon
    - expected_depth_rate: Expected depth change from pitch
    - depth_anomaly: Expected vs actual depth
    - heading_efficiency: Actual vs expected heading change
    - vertical_velocity: Depth rate of change
    - horizontal_velocity_magnitude: Speed from water velocity
    - course_over_ground: Actual direction of motion
    - current_estimation: Difference between heading and COG
    
    Args:
        df: Input DataFrame with sensor readings
        time_col: Name of time column
        
    Returns:
        DataFrame with added derived features
    """
    df = df.copy()
    
    # Constants for Slocum glider
    DEGREES_TO_RADIANS = np.pi / 180.0
    EARTH_RADIUS_KM = 6371.0
    
    # === POSITION & DISTANCE ===
    if 'm_lat' in df.columns and 'm_lon' in df.columns:
        # Calculate distance traveled
        lat_rad = df['m_lat'].values * DEGREES_TO_RADIANS
        lon_rad = df['m_lon'].values * DEGREES_TO_RADIANS
        
        dlat = np.diff(lat_rad, prepend=lat_rad[0])
        dlon = np.diff(lon_rad, prepend=lon_rad[0])
        
        # Haversine distance approximation
        a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(lat_rad.shift(1).fillna(lat_rad)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance_km = EARTH_RADIUS_KM * c
        distance_km = distance_km.fillna(0)
        
        df['distance_traveled'] = distance_km.cumsum()
        df['distance_traveled'] = df['distance_traveled'].fillna(0)
    
    # === VERTICAL VELOCITY ===
    if 'm_depth' in df.columns:
        df['vertical_velocity'] = df['m_depth'].diff().fillna(0)
    
    # === EXPECTED DEPTH RATE FROM PITCH ===
    # Slocum pitch angle → vertical velocity relationship
    # Typical pitch: -45° (diving) to +45° (climbing)
    if 'm_pitch' in df.columns:
        # Convert pitch to vertical velocity (approximate)
        # At 45° pitch, vertical velocity ≈ speed * sin(45°) ≈ speed * 0.707
        if 'm_speed' in df.columns:
            pitch_rad = df['m_pitch'].values * DEGREES_TO_RADIANS
            expected_vertical = df['m_speed'].values * np.sin(pitch_rad)
            df['expected_vertical_velocity'] = expected_vertical
            
            # Depth anomaly: expected vs actual
            if 'vertical_velocity' in df.columns:
                df['depth_anomaly'] = df['vertical_velocity'] - df['expected_vertical_velocity']
    
    # === HEADING EFFICIENCY ===
    if 'm_heading' in df.columns and 'm_roll' in df.columns:
        # Roll affects heading efficiency
        # High roll → less efficient propulsion → reduced effective heading change
        roll_rad = df['m_roll'].values * DEGREES_TO_RADIANS
        heading_efficiency = np.cos(roll_rad)  # cos(0) = 1 (efficient), cos(45°) ≈ 0.707
        df['heading_efficiency'] = heading_efficiency
    
    # === HORIZONTAL VELOCITY MAGNITUDE ===
    if 'm_final_water_vx' in df.columns and 'm_final_water_vy' in df.columns:
        vx = df['m_final_water_vx'].values
        vy = df['m_final_water_vy'].values
        df['horizontal_velocity_magnitude'] = np.sqrt(vx**2 + vy**2)
    
    # === COURSE OVER GROUND ===
    if 'm_final_water_vx' in df.columns and 'm_final_water_vy' in df.columns:
        vx = df['m_final_water_vx'].values
        vy = df['m_final_water_vy'].values
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            cog = np.arctan2(vy, vx) / DEGREES_TO_RADIANS
            cog = np.where(np.isnan(cog), 0, cog)
        df['course_over_ground'] = cog
    
    # === CURRENT ESTIMATION (heading vs COG) ===
    if 'm_heading' in df.columns and 'course_over_ground' in df.columns:
        heading = df['m_heading'].values
        cog = df['course_over_ground'].values
        
        # Angular difference accounting for wraparound
        diff = cog - heading
        diff = np.where(diff > 180, diff - 360, diff)
        diff = np.where(diff < -180, diff + 360, diff)
        df['current_estim_x'] = diff * np.sin(heading * DEGREES_TO_RADIANS)
        df['current_estim_y'] = diff * np.cos(heading * DEGREES_TO_RADIANS)
    
    # === ENERGY RATE ===
    if 'm_coulomb_amphr_total' in df.columns:
        df['energy_consumption_rate'] = df['m_coulomb_amphr_total'].diff().fillna(0)
    
    # Fill any NaN values
    df = df.fillna(0)
    
    return df


def create_feature_matrix(df: pd.DataFrame,
                         include_temporal: bool = True,
                         include_derived: bool = True,
                         time_col: str = 'time') -> Tuple[np.ndarray, List[str]]:
    """
    Create feature matrix with optional temporal and derived features.
    
    Args:
        df: Input DataFrame
        include_temporal: Add delta and rolling features
        include_derived: Add physics-based derived features
        time_col: Name of time column
        
    Returns:
        Tuple of (feature_matrix [T, n_features], feature_names)
    """
    if include_derived:
        df = compute_derived_features(df, time_col)
    
    if include_temporal:
        df = compute_temporal_features(df, time_col)
    
    # Get all feature columns (exclude metadata)
    exclude = ['year', 'julian_day', 'dive_cycle', time_col]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    feature_matrix = df[feature_cols].values.astype(np.float32)
    
    return feature_matrix, feature_cols


def save_graph_topology(filepath: str):
    """Save graph topology to JSON file."""
    topology = {
        'sensor_to_idx': SENSOR_TO_IDX,
        'idx_to_sensor': IDX_TO_SENSOR,
        'edges': [(SENSOR_TO_IDX[s], SENSOR_TO_IDX[t], w) 
                  for (s, t), w in GRAPH_EDGES.items()],
        'edge_descriptions': {f"{s}->{t}": desc 
                            for (s, t), desc in EDGE_DESCRIPTIONS.items()}
    }
    
    with open(filepath, 'w') as f:
        json.dump(topology, f, indent=2)
    
    print(f"Graph topology saved to {filepath}")


# ============================================================
# EDGE DESCRIPTIONS FOR DOCUMENTATION
# ============================================================

EDGE_DESCRIPTIONS = {
    ('m_heading', 'm_lat'): "Heading affects latitudinal trajectory",
    ('m_heading', 'm_lon'): "Heading affects longitudinal trajectory",
    ('m_speed', 'm_lat'): "Speed determines rate of lat change",
    ('m_speed', 'm_lon'): "Speed determines rate of lon change",
    ('m_pitch', 'm_depth'): "Pitch angle directly controls depth change rate",
    ('m_roll', 'm_heading'): "Roll affects heading efficiency",
    ('m_ballast_pumped', 'm_depth'): "Ballast changes buoyancy → depth",
    ('m_battery', 'm_battery_inst'): "Battery voltage relationship",
    ('m_battery', 'm_speed'): "Battery state affects available power",
    ('sci_water_temp', 'sci_water_pressure'): "Thermocline relationship",
    ('sci_water_temp', 'sci_water_cond'): "T-S relationship",
    ('m_depth', 'sci_water_temp'): "Depth affects water temperature",
    ('m_depth', 'sci_water_pressure'): "Pressure = f(depth)",
}


if __name__ == "__main__":
    # Test the module
    print("=== AUV Graph Topology ===")
    print(f"Number of sensors: {len(SENSOR_TO_IDX)}")
    print(f"Number of edges: {len(GRAPH_EDGES)}")
    
    A = build_adjacency_matrix()
    print(f"\nAdjacency matrix shape: {A.shape}")
    print(f"Non-zero edges: {np.count_nonzero(A)}")
    
    # Save topology
    save_graph_topology("graph_topology.json")
