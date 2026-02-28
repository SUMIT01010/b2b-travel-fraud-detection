# Walkthrough - Graph Matrix Generation

This document summarizes the changes made to generate Structural and Attribute matrices for the graph-based ML pipeline.

## Changes Made

### [NEW] [generate_graph_matrices.py](file:///Users/sumitsaurabh/fraud_detection/b2b-travel-fraud-detection/generate_graph_matrices.py)

A standalone script was created to process `booking_fact.csv`, `session_context.csv`, and `user_master.csv`.

**Key Features:**
- **Structural Matrix (A)**: Calculates connection weights between bookings using the formula:
  $W_{ij} = 0.6 \cdot M_{ip} + 0.4 \cdot e^{-0.05 |T_i - T_j|}$
  where $M_{ip}$ is IP match and $T$ is session duration.
- **Attribute Matrix (X)**: Extracts features (`is_vpn_or_proxy`, `device_switch_flag`, `avg_logins_per_day`, `failed_login_ratio`) and applies Min-Max normalization.
- **Efficiency**: Uses vectorized `numpy` operations to handle $8000 \times 8000$ matrices in memory.
- **Portability**: Uses relative paths to ensure the script works in any directory containing the `Data/` folder.

## Verification Results

### Matrix Generation
The script was executed successfully:
```bash
python3 generate_graph_matrices.py
```
**Output Logs:**
```
Loading data...
Number of unique bookings: 8000
Generating Structural Matrix...
Saved Structural Matrix to Data/graph/structural_matrix.csv
Generating Attribute Matrix...
Saved Attribute Matrix to Data/graph/attribute_matrix.csv
```

### File Verification
| Matrix Type | Dimensions | File Size | Range |
| :--- | :--- | :--- | :--- |
| **Structural (Adjacency)** | 8000 x 8000 | 1.3 GB | [0.0, 1.0] |
| **Attribute (Feature)** | 8000 x 4 | 432 KB | [0.0, 1.0] |

**Sample Data Check:**
- Diagonal of the Structural Matrix is `1.0`.
- All features in the Attribute Matrix were verified to be within the `[0.0, 1.0]` range.

## How to Run Again
If you need to regenerate the matrices, simply run:
```bash
cd /Users/sumitsaurabh/fraud_detection/b2b-travel-fraud-detection
python3 generate_graph_matrices.py
```
> [!NOTE]
> The Structural Matrix file is large (~1.3GB) due to the $8000^2$ entries being stored in CSV format. Ensure sufficient disk space.
