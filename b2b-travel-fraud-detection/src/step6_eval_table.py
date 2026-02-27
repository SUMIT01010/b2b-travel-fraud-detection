import numpy as np
import pandas as pd
import pickle
import glob

# -------------------------
# Load node map
# -------------------------

with open("data/graph/node_map.pkl","rb") as f:
    node_map = pickle.load(f)

# invert node_map: index -> node_id
inv_map = {v:k for k,v in node_map.items()}

# -------------------------
# Load DONE anomaly scores
# -------------------------

oval_files = sorted(glob.glob("DONE_AdONE/ovals/*b2b*"))

scores = []
for f in oval_files:
    scores.append(np.loadtxt(f))

scores = np.vstack(scores).T   # shape: N x num_scores
num_scores = scores.shape[1]

# -------------------------
# Build base node table
# -------------------------

rows = []

for i in range(scores.shape[0]):
    nid = inv_map[i]

    # derive node type from prefix
    if nid.startswith("A_"):
        nt = "agency"
    elif nid.startswith("U_"):
        nt = "user"
    elif nid.startswith("D_"):
        nt = "device"
    elif nid.startswith("I_"):
        nt = "ip"
    elif nid.startswith("B_"):
        nt = "booking"
    else:
        nt = "unknown"

    rows.append([nid, nt] + scores[i].tolist())

cols = ["node_id","node_type"] + [f"anomaly_score_{i+1}" for i in range(num_scores)]

df = pd.DataFrame(rows, columns=cols)

# -------------------------
# Load booking fraud labels + reasons
# -------------------------

booking_labels = pd.read_csv("data/booking_label_table.xls")

booking_labels["booking_node"] = booking_labels["booking_id"].astype(str).apply(lambda x: "B_"+x)

booking_flag_map = dict(
    zip(
        booking_labels.booking_node,
        zip(booking_labels.fraud_label, booking_labels.fraud_reason)
    )
)

# -------------------------
# Load user fraud labels + reasons
# -------------------------

user_labels = pd.read_csv("data/user_master.xls")

user_labels["user_node"] = user_labels["user_id"].astype(str).apply(lambda x: "U_"+x)

user_flag_map = dict(
    zip(
        user_labels.user_node,
        zip(user_labels.user_fraud_label, user_labels.user_fraud_type)
    )
)

# -------------------------
# Assign fraud flag + reason
# -------------------------

def get_flag_reason(n):
    if n.startswith("B_"):
        return booking_flag_map.get(n, (np.nan, np.nan))
    if n.startswith("U_"):
        return user_flag_map.get(n, (np.nan, np.nan))
    return (np.nan, np.nan)

df[["fraud_flag","fraud_reason"]] = pd.DataFrame(
    df.node_id.apply(get_flag_reason).tolist(),
    index=df.index
)

# -------------------------
# Save
# -------------------------

df.to_csv("node_evaluation_table.csv", index=False)

print("Saved node_evaluation_table.csv")
print(df.head())
print("\nNode counts:")
print(df.node_type.value_counts())
print("\nNon-null fraud flags:", df.fraud_flag.notna().sum())