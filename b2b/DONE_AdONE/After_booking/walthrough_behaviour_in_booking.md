# Walkthrough - Graph Matrix Generation (Optimized)

This document summarizes the logic and changes for generating the Structural and Attribute matrices used in the After-booking fraud detection pipeline.

## Graph Logic

The structural matrix $W$ is calculated using a combination of identity anchoring, behavioral similarity, and outcome reinforcement, all constrained by a temporal gate.

### 1. Structural Matrix (A)
The connection weight $W_{ij}$ between booking $i$ and booking $j$ is defined as:

$$W_{ij} = [ (f_{val} \cdot f_{lead}) + R_{ij} ] \cdot \text{TemporalGate} + A_{ij}$$

**Components:**
- **Temporal Gate**: Weights are only calculated if bookings occur within **7 days** of each other.
- **Identity Anchoring ($A_{ij}$)**: Direct weights for shared identifiers:
  - Shared Agency: +0.01
  - Shared User: +0.05
- **Similarity ($f_{val}, f_{lead}$)**: Gaussian decay based on booking value and lead time.
- **Outcome Reinforcement ($R_{ij}$)**: Shared negative outcomes increase the connection strength:
  - Both Cancelled: +0.15
  - Both Disputed: +0.20
  - Both Suspicious: +0.30
- **Constraint**: Diagonal is set to **0.0** (no self-loops).

### 2. Attribute Matrix (X)
Extracts 7 key features for each node:
- `lead_time_days`, `is_cancelled`, `cancel_delay_days`, `is_disputed`, `dispute_delay_days`, `chargeback_amount`, `final_loss_amount`.
- All features are **Min-Max Normalized** to the $[0, 1]$ range.

## Implementation Details

### Optimized Vectorization
The script [generate_graph_matrices_b.py](file:///Users/sumitsaurabh/fraud_detection/b2b/DONE_AdONE/After_booking/generate_graph_matrices_b.py) uses NumPy broadcasting to compute the $8000 \times 8000$ matrix. This replaces iterative row processing, reducing generation time from minutes to seconds.

## Verification Results

### Execution
```bash
python3 generate_graph_matrices_b.py
```

### Output Files
| Matrix Type | File Name | Dimensions | Format |
| :--- | :--- | :--- | :--- |
| **Structural** | `structural_matrix_b.csv` | 8000 x 8000 | Dense CSV (with IDs) |
| **Attribute** | `attribute_matrix_b.csv` | 8000 x 8 | Dense CSV (Node Features) |

> [!TIP]
> The matrix is stored in dense format to be directly compatible with the `AutoEncoder` model in `run_done_b.py`, which expects a standard NumPy array structure.
