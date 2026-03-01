import os
import numpy as np
import pandas as pd
from tqdm import tqdm

r"""
Script: generate_graph_matrices_b.py
Description: 
    This script implements the precise structural matrix math: using Temporal Gating (7 days), 
    booking value & lead time similarity ($f_{val} \cdot f_{lead}$), outcome reinforcement ($R_{ij}$), 
    and Identity Anchoring ($A_{ij}$), while leaving attributes constant. 
    It builds the edges iteratively to bypass RAM/Memory limitations for large dense matrices.
    
    Data Source: /Users/sumitsaurabh/fraud_detection/b2b/Data/Tables/master_table.csv
    Output Directory: /Users/sumitsaurabh/fraud_detection/b2b/Data/graph
"""

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(BASE_PATH, "Data", "Tables", "master_table.csv")
OUTPUT_DIR = os.path.join(BASE_PATH, "Data", "graph")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Weight parameters (as defined in the notebook)
ALPHA = 0.15      # Cancellation match weight
BETA = 0.20       # Dispute match weight
GAMMA = 0.30      # Suspicious match weight
W_AGENCY = 0.01   # Agency identity match weight
W_USER = 0.05     # User identity match weight
TEMPORAL_THRESHOLD = 7  # 7 days gating

# --- 1. Data Loading ---
print(f"Loading data from {DATA_PATH}...")
master_df = pd.read_csv(DATA_PATH)

# Derived indicators
master_df['is_suspicious'] = master_df['suspicious_pax_domains'] > 0

booking_ids = master_df['booking_id'].values
n_nodes = len(booking_ids)
print(f"Total Nodes (Bookings): {n_nodes}")

# Fill missing values
master_df = master_df.fillna(0)

# --- 2. Extract Features for Structural Calculation ---
# Convert timestamps to fractional days for temporal gating
t_arr = pd.to_datetime(master_df['booking_ts']).astype('int64').values / (10**9 * 60 * 60 * 24)

# Log-transform booking value
v_prime = np.log1p(master_df['booking_value'].values)

# Lead time days
l_arr = master_df['lead_time_days'].values

# Discrete indicators
cancel_arr = master_df['is_cancelled'].values
dispute_arr = master_df['is_disputed'].values
susp_arr = master_df['is_suspicious'].values.astype(bool)
agency_arr = np.array(master_df['agency_id'].values)
user_arr = np.array(master_df['user_id'].values)

# --- 3. Precompute Normalizations & Hyperparameters ---
# Normalization for booking value (v_tilde)
v_min, v_max = v_prime.min(), v_prime.max()
v_tilde = (v_prime - v_min) / (v_max - v_min + 1e-8)

# Compute lambda_val (Gaussian decay scale for value) using a subset to save memory
np.random.seed(42)
subset_size = min(3000, n_nodes)
sub_idx = np.random.choice(n_nodes, subset_size, replace=False)
v_sub = v_tilde[sub_idx]
delta_v_sub = np.abs(v_sub[:, None] - v_sub[None, :])
lambda_val = -np.log(0.2) / (np.percentile(delta_v_sub, 75) + 1e-8)
del delta_v_sub

# Normalization for lead time (l_tilde)
l_min, l_max = l_arr.min(), l_arr.max()
l_tilde = (l_arr - l_min) / (l_max - l_min + 1e-8)
l_sub = l_tilde[sub_idx]
delta_l_sub = np.abs(l_sub[:, None] - l_sub[None, :])
eta_val = -np.log(0.2) / (np.percentile(delta_l_sub, 75) + 1e-8)
del delta_l_sub

# --- 4. Dense Structural Matrix Calculation ---
print("Calculating dense structural matrix weights (vectorized)...")

# 1. Temporal Gating: Only consider connections within 7 days
# Using broadcasting for N x N comparison
t_diff = np.abs(t_arr[:, np.newaxis] - t_arr[np.newaxis, :])
temporal_mask = (t_diff <= TEMPORAL_THRESHOLD).astype(np.float32)

# 2. Identity Anchoring: Weight for common agency or user
agency_match = (agency_arr[:, np.newaxis] == agency_arr[np.newaxis, :])
user_match = (user_arr[:, np.newaxis] == user_arr[np.newaxis, :])
A_mat = (W_AGENCY * agency_match) + (W_USER * user_match)

# 3. Booking Similarity (Value & Lead Time)
v_diff = np.abs(v_tilde[:, np.newaxis] - v_tilde[np.newaxis, :])
f_val_mat = np.exp(-lambda_val * v_diff)

l_diff = np.abs(l_tilde[:, np.newaxis] - l_tilde[np.newaxis, :])
f_lead_mat = np.exp(-eta_val * l_diff)

# 4. Outcome Reinforcement: Weight for shared negative outcomes
c_match = ((cancel_arr[:, np.newaxis] == 1) & (cancel_arr[np.newaxis, :] == 1))
d_match = ((dispute_arr[:, np.newaxis] == 1) & (dispute_arr[np.newaxis, :] == 1))
s_match = ((susp_arr[:, np.newaxis] == True) & (susp_arr[np.newaxis, :] == True))
R_mat = (ALPHA * c_match) + (BETA * d_match) + (GAMMA * s_match)

# Combine terms
W_dense = ((f_val_mat * f_lead_mat) + R_mat) * temporal_mask + A_mat

# SET DIAGONAL TO ZERO: No self-loops
np.fill_diagonal(W_dense, 0.0)

# Save Structural Matrix (Dense Format)
structural_df = pd.DataFrame(W_dense, index=booking_ids, columns=booking_ids)
structural_output = os.path.join(OUTPUT_DIR, "structural_matrix_b.csv")
structural_df.to_csv(structural_output)
print(f"Saved dense structural matrix ({n_nodes}x{n_nodes}) to {structural_output}")

# --- 5. Attribute Matrix Formulation ---
# Features for the attribute matrix
attribute_cols = [
    'lead_time_days',
    'is_cancelled',
    'cancel_delay_days',
    'is_disputed',
    'dispute_delay_days',
    'chargeback_amount',
    'final_loss_amount'
]

print("Preparing attribute matrix...")
X_cont = master_df[attribute_cols].fillna(0).values

# Min-Max Normalization across columns
feature_min = X_cont.min(axis=0)
feature_max = X_cont.max(axis=0)
X_norm = (X_cont - feature_min) / (feature_max - feature_min + 1e-8)

# Save Attribute Matrix (Node Features)
nodes_df = pd.DataFrame(X_norm, columns=attribute_cols)
nodes_df.insert(0, 'booking_id', booking_ids)

attribute_output = os.path.join(OUTPUT_DIR, "attribute_matrix_b.csv")
nodes_df.to_csv(attribute_output, index=False)
print(f"Saved attribute matrix to {attribute_output}")

print("Graph preprocessing complete successfully!")
