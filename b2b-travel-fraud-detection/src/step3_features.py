import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

df = pd.read_pickle("data/processed/merged.pkl")

with open("data/graph/node_map.pkl","rb") as f:
    node_map = pickle.load(f)

with open("data/graph/node_type.pkl","rb") as f:
    node_type = pickle.load(f)

# -----------------------
# Booking features
# -----------------------

df["booking_ts"] = pd.to_datetime(df["booking_ts"])

df["hour"] = df["booking_ts"].dt.hour
df["dow"] = df["booking_ts"].dt.dayofweek

df["booking_value"] = np.log1p(df["booking_value"])

num_cols = [
    "booking_value","passengers_count","lead_time_days",
    "cancel_delay_days","dispute_delay_days",
    "chargeback_amount","final_loss_amount",
    "hour","dow"
]

bin_cols = ["is_cancelled","is_disputed","is_proxy"]

df[num_cols] = df[num_cols].fillna(0)
df[bin_cols] = df[bin_cols].fillna(0)

booking_feat = df[["booking_id"] + num_cols + bin_cols]

booking_feat.index = booking_feat.booking_id.map(lambda x: f"B_{x}")
booking_feat.drop("booking_id",axis=1,inplace=True)

# -----------------------
# Aggregations
# -----------------------

agency_feat = df.groupby("agency_id").agg(
    booking_count=("booking_id","count"),
    mean_booking_value=("booking_value","mean"),
    cancel_rate=("is_cancelled","mean"),
    dispute_rate=("is_disputed","mean"),
    avg_passengers=("passengers_count","mean"),
    agency_age_days=("agency_age_days","mean"),
    credit_limit=("credit_limit","mean")
)
agency_feat.index = agency_feat.index.map(lambda x: f"A_{x}")

user_feat = df.groupby("user_id").agg(
    booking_count=("booking_id","count"),
    mean_booking_value=("booking_value","mean"),
    failed_login_ratio=("failed_login_ratio","mean"),
    avg_logins_per_day=("avg_logins_per_day","mean"),
    user_age_days=("user_age_days","mean")
)
user_feat.index = user_feat.index.map(lambda x: f"U_{x}")

device_feat = df.groupby("device_id").agg(
    booking_count=("booking_id","count"),
    unique_users=("user_id","nunique")
)
device_feat.index = device_feat.index.map(lambda x: f"D_{x}")

ip_feat = df.groupby("ip_id").agg(
    booking_count=("booking_id","count"),
    unique_agencies=("agency_id","nunique"),
    proxy_rate=("is_proxy","mean")
)
ip_feat.index = ip_feat.index.map(lambda x: f"I_{x}")

# -----------------------
# Combine all
# -----------------------

all_feat = pd.concat([
    booking_feat,
    agency_feat,
    user_feat,
    device_feat,
    ip_feat
], axis=0).fillna(0)

# Align to node_map order

X = np.zeros((len(node_map), all_feat.shape[1]))

for k,v in node_map.items():
    if k in all_feat.index:
        X[v] = all_feat.loc[k].values

# Standardize

scaler = StandardScaler()
X = scaler.fit_transform(X)

# np.save("data/graph/features.npy", X)

print("Saved features.npy with shape:", X.shape)

print(X)