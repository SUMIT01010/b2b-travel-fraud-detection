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

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)

# Compute lambda_val (Gaussian decay scale for value) using a subset to save memory
set_seed(42)
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

# --- 4. Structural Matrix Calculation (Chunked for Memory Efficiency) ---
print("Calculating structural matrix weights in chunks...")

chunk_size = 1000
structural_output = os.path.join(OUTPUT_DIR, "structural_matrix_b.csv")

with open(structural_output, 'w') as f:
    # Write header
    header = ",".join(booking_ids)
    f.write("," + header + "\n")

    for i in range(0, n_nodes, chunk_size):
        end_i = min(i + chunk_size, n_nodes)
        print(f"  Processing structural rows {i} to {end_i}...")
        
        # Slices for this chunk
        t_i = t_arr[i:end_i, np.newaxis]
        agency_i = agency_arr[i:end_i, np.newaxis]
        user_i = user_arr[i:end_i, np.newaxis]
        v_i = v_tilde[i:end_i, np.newaxis]
        l_i = l_tilde[i:end_i, np.newaxis]
        cancel_i = cancel_arr[i:end_i, np.newaxis]
        dispute_i = dispute_arr[i:end_i, np.newaxis]
        susp_i = susp_arr[i:end_i, np.newaxis]

        # 1. Temporal Gating
        t_diff_chunk = np.abs(t_i - t_arr)
        temporal_mask_chunk = (t_diff_chunk <= TEMPORAL_THRESHOLD).astype(np.float32)

        # 2. Identity Anchoring
        agency_match_chunk = (agency_i == agency_arr)
        user_match_chunk = (user_i == user_arr)
        A_chunk = (W_AGENCY * agency_match_chunk) + (W_USER * user_match_chunk)

        # 3. Booking Similarity
        v_diff_chunk = np.abs(v_i - v_tilde)
        f_val_chunk = np.exp(-lambda_val * v_diff_chunk)
        l_diff_chunk = np.abs(l_i - l_tilde)
        f_lead_chunk = np.exp(-eta_val * l_diff_chunk)

        # 4. Outcome Reinforcement
        c_match_chunk = ((cancel_i == 1) & (cancel_arr == 1))
        d_match_chunk = ((dispute_i == 1) & (dispute_arr == 1))
        s_match_chunk = ((susp_i == True) & (susp_arr == True))
        R_chunk = (ALPHA * c_match_chunk) + (BETA * d_match_chunk) + (GAMMA * s_match_chunk)

        # Combine terms for this chunk
        W_chunk = ((f_val_chunk * f_lead_chunk) + R_chunk) * temporal_mask_chunk + A_chunk

        # Ensure diagonal = 0
        for local_idx in range(end_i - i):
            global_idx = i + local_idx
            W_chunk[local_idx, global_idx] = 0.0

        # Save chunk to file
        batch_df = pd.DataFrame(W_chunk, index=booking_ids[i:end_i], columns=booking_ids)
        batch_df.to_csv(f, header=False)

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
