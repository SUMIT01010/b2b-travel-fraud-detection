import pandas as pd
import os

def create_bi_master_table(master_path, before_path, after_path, output_dir):
    print("Loading datasets...")
    try:
        master_df = pd.read_csv(master_path)
        before_df = pd.read_csv(before_path)
        after_df = pd.read_csv(after_path)
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        return

    # 1. Filter to keep ONLY booking_id, max_score, and predicted_fraud
    cols_to_keep = ['booking_id', 'max_score', 'predicted_fraud']
    
    # Check if all required columns exist in both dataframes
    for col in cols_to_keep:
        if col not in before_df.columns or col not in after_df.columns:
            print(f"❌ Error: Column '{col}' is missing from the prediction files.")
            return

    before_sub = before_df[cols_to_keep].copy()
    after_sub = after_df[cols_to_keep].copy()

    # 2. Rename columns to make them explicit for BI tools and prevent collisions
    before_sub.rename(columns={
        'max_score': 'max_score_before', 
        'predicted_fraud': 'predicted_fraud_before'
    }, inplace=True)
    
    after_sub.rename(columns={
        'max_score': 'max_score_after', 
        'predicted_fraud': 'predicted_fraud_after'
    }, inplace=True)

    # 3. Merge Tables (Left join keeps all master_table rows intact)
    print("Merging data...")
    merged_df = pd.merge(master_df, before_sub, on='booking_id', how='left')
    final_bi_df = pd.merge(merged_df, after_sub, on='booking_id', how='left')

    # 4. Save Output
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'b2b_bi_master.csv')
    final_bi_df.to_csv(output_file_path, index=False)
    
    print(f"\n✅ Created/Verified output folder at: {output_dir}")
    print(f"✅ Final BI table saved to: {output_file_path}")
    print(f"Total Rows: {len(final_bi_df)} | Total Columns: {len(final_bi_df.columns)}")

if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION: Exact folder paths
    # ==========================================
    
    MASTER_CSV_PATH = "/Users/shashwata/Documents/Data Science Projects/b2b-travel-fraud-detection/b2b/Data/Tables/master_table.csv"
    BEFORE_CSV_PATH = "/Users/shashwata/Documents/Data Science Projects/b2b-travel-fraud-detection/b2b/DONE_AdONE/Before_Booking/output/final_scored_predictions_before_booking.csv"
    AFTER_CSV_PATH  = "/Users/shashwata/Documents/Data Science Projects/b2b-travel-fraud-detection/b2b/DONE_AdONE/After_booking/output/final_scored_predictions_after_booking.csv"
    
    # Define where you want the new 'b2b' folder to be created
    OUTPUT_B2B_FOLDER = "/Users/shashwata/Documents/Data Science Projects/b2b-travel-fraud-detection/b2b/output_layer"
    
    # Run the function
    create_bi_master_table(
        master_path=MASTER_CSV_PATH, 
        before_path=BEFORE_CSV_PATH, 
        after_path=AFTER_CSV_PATH, 
        output_dir=OUTPUT_B2B_FOLDER
    )