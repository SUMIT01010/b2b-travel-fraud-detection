import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def generate_report_from_saved_scores(
    input_csv_path: str, 
    output_md_path: str,
    output_folder_path: str,
    score_col: str = 'max_score', 
    label_col: str = 'fraud_label'
):
    """
    Loads a saved CSV containing anomaly scores and labels, finds the optimal 
    threshold to maximize F1-score, adds the prediction flag, saves the final table 
    to an 'output' folder, and exports a classification report to Markdown.
    """
    print(f"Loading data from: {input_csv_path}")
    
    if not os.path.exists(input_csv_path):
        print(f"Error: Could not find the file at {input_csv_path}")
        return
        
    df = pd.read_csv(input_csv_path)
    
    # 1. Extract ground truth and scores
    if label_col not in df.columns or score_col not in df.columns:
        raise ValueError(f"The CSV must contain '{label_col}' and '{score_col}' columns.")
        
    y_true = df[label_col].fillna(0).astype(int).values
    scores = df[score_col].values
    
    print("\nSweeping percentiles (80% to 99.9%) for optimal threshold...")
    # 2. Sweep Thresholds (F1 Optimization)
    percentiles = np.linspace(80, 99.9, 200)
    
    best_f1 = -1
    best_thresh = None
    best_pct = None
    
    for p in percentiles:
        thresh = np.percentile(scores, p)
        y_pred = (scores > thresh).astype(int)
        
        if y_pred.sum() == 0:
            continue
            
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_pct = p
            
    print(f"Optimal Split -> Percentile: {best_pct:.2f}% | Threshold: {best_thresh:.6f} | F1: {best_f1:.4f}")
    
    # 3. Final Predictions & Saving Final Table
    y_pred_final = (scores > best_thresh).astype(int)
    
    # Add the prediction flag to the dataframe
    df['predicted_fraud'] = y_pred_final
    
    # Create the 'output' folder if it doesn't exist and save the final table
    os.makedirs(output_folder_path, exist_ok=True)
    final_output_csv = os.path.join(output_folder_path, 'final_scored_predictions_before_booking.csv')
    df.to_csv(final_output_csv, index=False)
    print(f"✅ Final predictions table saved to: {final_output_csv}")
    
    # 4. Metrics Calculation
    cm = confusion_matrix(y_true, y_pred_final)
    cr = classification_report(y_true, y_pred_final, digits=4)
    
    # Prepare Precision@K and Capture Rate data
    sorted_indices = scores.argsort()[::-1]
    sorted_y_true = y_true[sorted_indices]
    total_fraud = y_true.sum()
    
    metrics_k = []
    for k in [1, 2, 5, 10, 15]:
        topk = int(len(df) * k / 100)
        if topk > 0:
            fraud_captured = sorted_y_true[:topk].sum()
            prec_k = fraud_captured / topk
            recall_k = (fraud_captured / total_fraud) * 100
            metrics_k.append((k, topk, int(fraud_captured), prec_k, recall_k))

    # 5. Markdown Output Generation
    os.makedirs(os.path.dirname(os.path.abspath(output_md_path)), exist_ok=True)
    
    with open(output_md_path, 'w') as f:
        f.write("# Model Classification & Performance Report\n\n")
        
        f.write("## 1. Optimal Threshold Setup\n")
        f.write("By sweeping percentiles from 80% to 99.9%, the model identified the optimal split to maximize the F1-Score:\n")
        f.write(f"- **Optimal Percentile Split:** `{best_pct:.2f}%`\n")
        f.write(f"- **Score Threshold:** `{best_thresh:.6f}`\n")
        f.write(f"- **Achieved F1-Score:** `{best_f1:.4f}`\n\n")
        
        f.write("## 2. Confusion Matrix\n")
        f.write("```text\n")
        f.write(f"[[{cm[0,0]:<5} {cm[0,1]}]\n")
        f.write(f" [{cm[1,0]:<5} {cm[1,1]}]]\n")
        f.write("```\n")
        f.write(f"- **True Negatives (Legit properly classified):** {cm[0,0]}\n")
        f.write(f"- **False Positives (Legit flagged as fraud):** {cm[0,1]}\n")
        f.write(f"- **False Negatives (Fraud missed):** {cm[1,0]}\n")
        f.write(f"- **True Positives (Fraud caught):** {cm[1,1]}\n\n")
        
        f.write("## 3. Classification Report\n")
        f.write("```text\n")
        f.write(cr)
        f.write("\n```\n\n")
        
        f.write("## 4. Performance @ K%\n")
        f.write("Precision and capture rate (recall) if we manually flag the top K% of highest-scoring records.\n\n")
        f.write("| Top % | Records Checked | Fraud Found | Precision | Capture Rate (Recall) |\n")
        f.write("|-------|-----------------|-------------|-----------|-----------------------|\n")
        for k, topk, fraud_captured, prec_k, recall_k in metrics_k:
            f.write(f"| **{k}%** | {topk} | {fraud_captured} | {prec_k:.4f} | {recall_k:.2f}% |\n")
            
    print(f"✅ Report successfully saved to: {output_md_path}")


# ==========================================
# How to call it in your pipeline
# ==========================================
if __name__ == "__main__":
    # Define your paths 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_tables_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'Data', 'Tables')
    
    # 1. Point this to your newly saved CSV
    saved_csv_path = os.path.join(data_tables_path, 'scored_booking_labels_before_booking.csv')
    
    # 2. Define where you want the markdown report to go
    report_md_path = os.path.join(current_dir, 'classification_report.md')
    
    # 3. Define the new 'output' directory for the final prediction table
    output_dir = os.path.join(current_dir, 'output')
    
    # 4. Call the function
    generate_report_from_saved_scores(
        input_csv_path=saved_csv_path,
        output_md_path=report_md_path,
        output_folder_path=output_dir
    )