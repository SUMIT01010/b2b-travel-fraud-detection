# Model Classification & Performance Report

## 1. Optimal Threshold Setup
By sweeping percentiles from 80% to 99.9%, the model identified the optimal split to maximize the F1-Score:
- **Optimal Percentile Split:** `91.30%`
- **Score Threshold:** `0.000282`
- **Achieved F1-Score:** `0.2500`

## 2. Confusion Matrix
```text
[[6761  519]
 [543   177]]
```
- **True Negatives (Legit properly classified):** 6761
- **False Positives (Legit flagged as fraud):** 519
- **False Negatives (Fraud missed):** 543
- **True Positives (Fraud caught):** 177

## 3. Classification Report
```text
              precision    recall  f1-score   support

           0     0.9257    0.9287    0.9272      7280
           1     0.2543    0.2458    0.2500       720

    accuracy                         0.8672      8000
   macro avg     0.5900    0.5873    0.5886      8000
weighted avg     0.8652    0.8672    0.8662      8000

```

## 4. Performance @ K%
Precision and capture rate (recall) if we manually flag the top K% of highest-scoring records.

| Top % | Records Checked | Fraud Found | Precision | Capture Rate (Recall) |
|-------|-----------------|-------------|-----------|-----------------------|
| **1%** | 80 | 33 | 0.4125 | 4.58% |
| **2%** | 160 | 56 | 0.3500 | 7.78% |
| **5%** | 400 | 104 | 0.2600 | 14.44% |
| **10%** | 800 | 182 | 0.2275 | 25.28% |
| **15%** | 1200 | 220 | 0.1833 | 30.56% |
