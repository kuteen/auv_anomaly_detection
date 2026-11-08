"""
Enhanced Graph Topology for AUV Sensor Networks

Based on Slocum G2 glider domain knowledge:
- Navigation physics: lat/lon change based on heading + speed
- Vehicle dynamics: pitch → depth rate, roll → heading efficiency  
- Energy: battery state affects all operations
- Science: water properties are spatially correlated

Author: Enhanced for AUV Anomaly Detection
"""

import numpy as np
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
