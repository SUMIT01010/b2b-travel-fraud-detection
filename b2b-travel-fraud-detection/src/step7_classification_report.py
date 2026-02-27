import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# -----------------------
# Load evaluation table
# -----------------------

df = pd.read_csv("node_evaluation_table.csv")
print(df.shape)

# Only bookings & users (clean ground truth)
df = df[df.node_type.isin(["booking", "user"])].copy()

# Average DONE scores
df["final_score"] = df[["anomaly_score_1","anomaly_score_2","anomaly_score_3"]].mean(axis=1)

# Ground truth
y_true = df["fraud_flag"].fillna(0).astype(int).values
scores = df["final_score"].values

# -----------------------
# Sweep thresholds
# -----------------------

percentiles = np.linspace(80, 99.9, 200)

best_f1 = -1
best_thresh = None
best_pct = None

for p in percentiles:
    thresh = np.percentile(scores, p)
    y_pred = (scores > thresh).astype(int)

    if y_pred.sum() == 0:
        continue

    f1 = f1_score(y_true, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        best_pct = p

print("\nBest threshold found:")
print("Percentile:", round(best_pct,2))
print("Score threshold:", best_thresh)
print("Best F1:", round(best_f1,4))

# -----------------------
# Final prediction
# -----------------------

df["predicted"] = (scores > best_thresh).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, df.predicted))

print("\nClassification Report:")
print(classification_report(y_true, df.predicted, digits=4))

# -----------------------
# Precision@K
# -----------------------

for k in [1,2,5,10]:
    topk = int(len(df) * k / 100)
    prec_k = y_true[df.final_score.argsort()[::-1][:topk]].mean()
    print(f"Precision@{k}%:", round(prec_k,4))
