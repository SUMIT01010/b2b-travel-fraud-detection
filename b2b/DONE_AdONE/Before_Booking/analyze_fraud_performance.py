import pandas as pd
import numpy as np
import os

def analyze_performance():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ovals_dir = os.path.join(current_dir, 'ovals')
    data_tables_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'Data', 'Tables')
    
    # Load Oval Scores
    print("Loading oval scores...")
    o1 = np.loadtxt(os.path.join(ovals_dir, 'b2b_done-oval1'))
    o2 = np.loadtxt(os.path.join(ovals_dir, 'b2b_done-oval2'))
    o3 = np.loadtxt(os.path.join(ovals_dir, 'b2b_done-oval3'))
    
    # Load Labels
    label_file = os.path.join(data_tables_path, 'booking_label_table.csv')
    labels_df = pd.read_csv(label_file)
    
    # Ensure alignment (8000 records)
    # The oval scores match the order in master_table, which matches booking_label_table (skipping header)
    # but let's be safe and assume they are in the same order as booking_id B000001 to B008000
    
    print(f"Number of scores: {len(o1)}")
    print(f"Number of labels: {len(labels_df)}")
    
    # Calculate Max Oval
    max_scores = np.maximum(o1, np.maximum(o2, o3))
    
    # Combine into a single DataFrame for analysis
    # Assuming the order is consistent with the booking_id sequence in labels_df
    analysis_df = pd.DataFrame({
        'max_score': max_scores,
        'fraud_label': labels_df['fraud_label'].values[:len(max_scores)]
    })
    
    # Sort by score descending
    analysis_df = analysis_df.sort_values(by='max_score', ascending=False).reset_index(drop=True)
    
    total_records = len(analysis_df)
    total_fraud = analysis_df['fraud_label'].sum()
    
    print(f"\nTotal Records: {total_records}")
    print(f"Total Fraud Cases: {total_fraud}")
    print("\nCapture Analysis (Top X% of data):")
    print("-" * 50)
    print(f"{'Top %':<10} | {'Records':<10} | {'Fraud Captured':<15} | {'Capture %'}")
    print("-" * 50)
    
    thresholds = [0.01, 0.05, 0.10, 0.15]
    for pct in thresholds:
        num_records = int(pct * total_records)
        fraud_in_top = analysis_df.iloc[:num_records]['fraud_label'].sum()
        capture_pct = (fraud_in_top / total_fraud) * 100
        
        print(f"{int(pct*100):>4}%      | {num_records:>9}  | {int(fraud_in_top):>14}  | {capture_pct:.2f}%")
    print("-" * 50)

    # Update performance.md
    perf_file = os.path.join(current_dir, 'performance.md')
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(perf_file, 'w') as f:
        f.write("# Fraud Performance Analysis\n\n")
        f.write("The following results evaluate the fraud detection model's performance by calculating the maximum of the three \"oval\" (outlier) scores and determining the fraud capture rate at various population thresholds.\n\n")
        f.write("## Performance Results\n\n")
        f.write("| Top % | Records | Fraud Captured | Capture % |\n")
        f.write("|-------|---------|----------------|-----------|\n")
        for pct in thresholds:
            num_records = int(pct * total_records)
            fraud_in_top = analysis_df.iloc[:num_records]['fraud_label'].sum()
            capture_pct = (fraud_in_top / total_fraud) * 100
            f.write(f"| **{int(pct*100)}%** | {num_records} | {int(fraud_in_top)} | {capture_pct:.2f}% |\n")
        
        f.write(f"\nThe results show that the model captures **{analysis_df.iloc[:int(0.1*total_records)]['fraud_label'].sum() / total_fraud * 100:.2f}% of all fraud cases within the top 10%** of the highest-scored bookings.\n\n")
        f.write(f"*Last updated: {now}*")
    
    print(f"Updated {perf_file}")

if __name__ == "__main__":

    analyze_performance()
