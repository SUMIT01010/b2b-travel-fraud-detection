import pandas as pd
import numpy as np
import os

def generate_matrices():
    # Define paths relative to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If script is inside DONE_AdONE, the root is one level up
    if os.path.basename(current_dir) == 'DONE_AdONE':
        base_path = os.path.dirname(current_dir)
    else:
        base_path = current_dir
        
    data_tables_path = os.path.join(base_path, 'Data', 'Tables')
    output_path = os.path.join(base_path, 'Data', 'graph')
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    print("Loading data...")
    booking_fact = pd.read_csv(os.path.join(data_tables_path, 'booking_fact.csv'))
    session_context = pd.read_csv(os.path.join(data_tables_path, 'session_context.csv'))
    user_master = pd.read_csv(os.path.join(data_tables_path, 'user_master.csv'))
    
    # Preprocessing: Map time_to_book_seconds to session_duration
    session_context = session_context.rename(columns={'time_to_book_seconds': 'session_duration'})
    
    # Join Data
    # Join session_context -> booking_fact -> user_master
    df = session_context.merge(booking_fact, on='booking_id', how='left')
    df = df.merge(user_master, on='user_id', how='left')
    
    # Handle missing values if any
    df['avg_logins_per_day'] = df['avg_logins_per_day'].fillna(0)
    df['failed_login_ratio'] = df['failed_login_ratio'].fillna(0)
    df['is_vpn_or_proxy'] = df['is_vpn_or_proxy'].fillna(0).astype(int)
    df['device_switch_flag'] = df['device_switch_flag'].fillna(0).astype(int)
    
    booking_ids = df['booking_id'].values
    n = len(booking_ids)
    print(f"Number of unique bookings: {n}")
    
    # --- 1. Structural Matrix (Adjacency Matrix A) ---
    print("Generating Structural Matrix...")
    device_fingerprints = np.array(df['device_fingerprint'])
    ip_addresses = np.array(df['ip_address'])
    durations = np.array(df['session_duration'])
    
    # Vectorized calculation for W_ij
    # M_dev: 1 if device_fingerprint matches, 0 otherwise
    # M_ip: 1 if ip_address matches, 0 otherwise
    # Weight Formula: W_ij = 0.5 * M_dev + 0.3 * M_ip + 0.2 * e^(-0.05 * |T_i - T_j|)
    
    # Using broadcasting for efficient computation
    # M_dev matrix
    m_dev = (device_fingerprints[:, np.newaxis] == device_fingerprints).astype(float)
    
    # M_ip matrix
    m_ip = (ip_addresses[:, np.newaxis] == ip_addresses).astype(float)
    
    # T_diff matrix
    t_diff = np.abs(durations[:, np.newaxis] - durations)
    
    gamma = 0.05
    w = 0.5 * m_dev + 0.3 * m_ip + 0.2 * np.exp(-gamma * t_diff)
    
    structural_df = pd.DataFrame(w, index=booking_ids, columns=booking_ids)
    structural_file = os.path.join(output_path, 'structural_matrix.csv')
    structural_df.to_csv(structural_file)
    print(f"Saved Structural Matrix to {structural_file}")
    
    # --- 2. Attribute Matrix (Feature Matrix X) ---
    print("Generating Attribute Matrix...")
    features = df[['is_vpn_or_proxy', 'device_switch_flag', 'avg_logins_per_day', 'failed_login_ratio']].copy()
    
    # Manual Normalization (MinMaxScaler)
    for col in ['avg_logins_per_day', 'failed_login_ratio']:
        min_val = features[col].min()
        max_val = features[col].max()
        if max_val > min_val:
            features[col] = (features[col] - min_val) / (max_val - min_val)
        else:
            features[col] = 0.0
    
    attribute_df = pd.DataFrame(features.values, index=booking_ids, columns=features.columns)
    attribute_file = os.path.join(output_path, 'attribute_matrix.csv')
    attribute_df.to_csv(attribute_file)
    print(f"Saved Attribute Matrix to {attribute_file}")

if __name__ == "__main__":
    generate_matrices()
