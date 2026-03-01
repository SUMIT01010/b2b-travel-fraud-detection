import pandas as pd
import os

def create_master_table(data_dir):
    """
    Consolidates data from all individual tables into a single master table keyed by booking_id.
    """
    print(f"Loading data from {data_dir}...")
    
    # Load all tables
    booking_fact = pd.read_csv(os.path.join(data_dir, "booking_fact.csv"))
    booking_label = pd.read_csv(os.path.join(data_dir, "booking_label_table.csv"))
    session_context = pd.read_csv(os.path.join(data_dir, "session_context.csv"))
    passenger_details = pd.read_csv(os.path.join(data_dir, "passenger_details.csv"))
    post_booking_events = pd.read_csv(os.path.join(data_dir, "post_booking_events.csv"))
    user_master = pd.read_csv(os.path.join(data_dir, "user_master.csv"))
    agency_master = pd.read_csv(os.path.join(data_dir, "agency_master.csv"))

    print("Processing passenger details...")
    # Aggregate passenger details (e.g., number of passengers, number of suspicious domains)
    pax_agg = passenger_details.groupby('booking_id').agg(
        total_passengers=('passenger_id', 'count'),
        suspicious_pax_domains=('is_suspicious_domain', 'sum')
    ).reset_index()

    print("Joining tables...")
    # Start with booking_fact and join everything else
    master = booking_fact.merge(booking_label, on='booking_id', how='left')
    master = master.merge(session_context, on='booking_id', how='left')
    master = master.merge(post_booking_events, on='booking_id', how='left')
    master = master.merge(pax_agg, on='booking_id', how='left')
    
    # Join user_master (via user_id)
    # Prefix user columns to avoid collisions, keeping user_id as key
    user_cols = ['user_id', 'role', 'user_age_days', 'avg_logins_per_day', 
                 'failed_login_ratio', 'account_status', 'email_domain_match_flag', 
                 'user_fraud_label', 'user_fraud_type']
    master = master.merge(user_master[user_cols], on='user_id', how='left', suffixes=('', '_user'))

    # Join agency_master (via agency_id)
    agency_cols = ['agency_id', 'country', 'agency_age_days', 'kyc_status', 
                   'credit_limit', 'status', 'agency_email_domain']
    master = master.merge(agency_master[agency_cols], on='agency_id', how='left', suffixes=('', '_agency'))

    output_path = os.path.join(data_dir, "master_table.csv")
    master.to_csv(output_path, index=False)
    print(f"✅ Success! Master table created at: {output_path}")
    print(f"Total rows: {len(master)}")
    print(f"Total columns: {len(master.columns)}")

if __name__ == "__main__":
    # Assuming the standard directory structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "Data", "Tables")
    
    if os.path.exists(target_dir):
        create_master_table(target_dir)
    else:
        print(f"Error: Directory not found - {target_dir}")
