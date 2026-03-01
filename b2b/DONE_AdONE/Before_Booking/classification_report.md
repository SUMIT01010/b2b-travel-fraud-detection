# Model Classification & Performance Report

## 1. Optimal Threshold Setup
By sweeping percentiles from 80% to 99.9%, the model identified the optimal split to maximize the F1-Score:
- **Optimal Percentile Split:** `91.50%`
- **Score Threshold:** `0.000690`
- **Achieved F1-Score:** `0.7971`

## 2. Confusion Matrix
```text
[[7158  122]
 [162   558]]
```
- **True Negatives (Legit properly classified):** 7158
- **False Positives (Legit flagged as fraud):** 122
- **False Negatives (Fraud missed):** 162
- **True Positives (Fraud caught):** 558

## 3. Classification Report
```text
              precision    recall  f1-score   support

           0     0.9779    0.9832    0.9805      7280
           1     0.8206    0.7750    0.7971       720

    accuracy                         0.9645      8000
   macro avg     0.8992    0.8791    0.8888      8000
weighted avg     0.9637    0.9645    0.9640      8000

```

## 4. Performance @ K%
Precision and capture rate (recall) if we manually flag the top K% of highest-scoring records.

| Top % | Records Checked | Fraud Found | Precision | Capture Rate (Recall) |
|-------|-----------------|-------------|-----------|-----------------------|
| **1%** | 80 | 80 | 1.0000 | 11.11% |
| **2%** | 160 | 159 | 0.9938 | 22.08% |
| **5%** | 400 | 368 | 0.9200 | 51.11% |
| **10%** | 800 | 577 | 0.7212 | 80.14% |
| **15%** | 1200 | 641 | 0.5342 | 89.03% |
