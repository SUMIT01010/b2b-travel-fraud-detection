# Fraud Performance Analysis

The following results evaluate the fraud detection model's performance by calculating the maximum of the three "oval" (outlier) scores and determining the fraud capture rate at various population thresholds.

## Performance Results

| Top % | Records | Fraud Captured | Capture % |
|-------|---------|----------------|-----------|
| **1%** | 80 | 80 | 11.11% |
| **5%** | 400 | 368 | 51.11% |
| **10%** | 800 | 577 | 80.14% |
| **15%** | 1200 | 641 | 89.03% |

The results show that the model captures **80.14% of all fraud cases within the top 10%** of the highest-scored bookings.

*Last updated: 2026-03-01 13:20:39*