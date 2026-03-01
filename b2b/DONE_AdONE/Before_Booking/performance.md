# Fraud Performance Analysis

The following results evaluate the fraud detection model's performance by calculating the maximum of the three "oval" (outlier) scores and determining the fraud capture rate at various population thresholds.

## Performance Results

| Top % | Records | Fraud Captured | Capture % |
|-------|---------|----------------|-----------|
| **1%** | 80 | 79 | 10.97% |
| **5%** | 400 | 335 | 46.53% |
| **10%** | 800 | 543 | 75.42% |
| **15%** | 1200 | 629 | 87.36% |

The results show that the model captures **75.42% of all fraud cases within the top 10%** of the highest-scored bookings.

*Last updated: 2026-03-01 04:23:23*