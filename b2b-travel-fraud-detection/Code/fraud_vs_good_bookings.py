import pandas as pd
import numpy as np
import os
import argparse

def analyze_booking_fraud(data_dir=None):
    """
    Analyzes and compares fraud bookings vs good baseline based on the generated synthetic dataset.
    
    Args:
        data_dir (str, optional): Directory containing the CSV files. 
                                  Defaults to ../Data/Tables relative to the script.
    """
    if data_dir is None:
        # Default: ../Data/Tables relative to this script
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "Data", "Tables")
    
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        print(f"❌ Error: Data directory not found: {data_dir}")
        return

    # -----------------------------
    # Load tables
    # -----------------------------
    try:
        users  = pd.read_csv(os.path.join(data_dir, "user_master.csv"))
        book   = pd.read_csv(os.path.join(data_dir, "booking_fact.csv"))
        label  = pd.read_csv(os.path.join(data_dir, "booking_label_table.csv"))
        events = pd.read_csv(os.path.join(data_dir, "post_booking_events.csv"))
        sess   = pd.read_csv(os.path.join(data_dir, "session_context.csv"))
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find one of the CSV files in {data_dir}: {e}")
        return

    # -----------------------------
    # Combined booking view
    # -----------------------------
    df = (book.merge(label, on="booking_id", how="left")
              .merge(events, on="booking_id", how="left")
              .merge(users[["user_id", "user_fraud_label", "user_fraud_type"]], on="user_id", how="left")
              .merge(sess[["booking_id", "device_fingerprint", "ip_address", "is_vpn_or_proxy"]], on="booking_id", how="left"))

    df["proxy_flag"] = df["is_vpn_or_proxy"].fillna(0).astype(int)

    fraud_df = df[df["fraud_label"] == 1].copy()
    good_df  = df[df["fraud_label"] == 0].copy()

    print(f"📂 Analyzing data from: {data_dir}")
    print(f"✅ Fraud bookings: {len(fraud_df)}")
    print(f"✅ Good bookings : {len(good_df)}")

    # -----------------------------
    # Helper: normalized infra metrics inside a subset
    # -----------------------------
    def normalized_infra(sub):
        per_user = (sub.groupby("user_id")
                      .agg(
                          total_bookings=("booking_id", "count"),
                          unique_ips=("ip_address", "nunique"),
                          unique_devices=("device_fingerprint", "nunique"),
                      )
                      .reset_index())

        per_user["ip_churn_rate"] = per_user["unique_ips"] / per_user["total_bookings"]
        per_user["device_churn_rate"] = per_user["unique_devices"] / per_user["total_bookings"]

        return {
            "avg_unique_ips_per_user": per_user["unique_ips"].mean(),
            "avg_unique_devices_per_user": per_user["unique_devices"].mean(),
            "avg_ip_churn_rate": per_user["ip_churn_rate"].mean(),
            "avg_device_churn_rate": per_user["device_churn_rate"].mean(),
        }

    # -----------------------------
    # Build FRAUD summary per fraud_reason
    # -----------------------------
    fraud_rows = []
    for reason, sub in fraud_df.groupby("fraud_reason"):
        infra = normalized_infra(sub)

        row = {
            "fraud_reason": reason,
            "group": "FRAUD",
            "num_bookings": len(sub),

            "avg_booking_value": sub["booking_value"].mean(),
            "avg_lead_time_days": sub["lead_time_days"].mean(),

            "cancellation_rate_pct": sub["is_cancelled"].mean() * 100,
            "avg_cancellation_delay_days": sub.loc[sub["is_cancelled"] == 1, "cancel_delay_days"].mean(),

            "dispute_rate_pct": sub["is_disputed"].mean() * 100,
            "avg_chargeback_amount": sub.loc[sub["is_disputed"] == 1, "chargeback_amount"].mean(),
            "avg_loss_amount": sub.loc[sub["is_disputed"] == 1, "final_loss_amount"].mean(),

            "proxy_usage_rate_pct": sub["proxy_flag"].mean() * 100,
            "international_booking_rate_pct": (sub["route_type"] == "international").mean() * 100,
        }
        row.update(infra)
        fraud_rows.append(row)

    fraud_summary = pd.DataFrame(fraud_rows)

    # -----------------------------
    # GOOD baseline (computed ONCE)
    # -----------------------------
    good_infra = normalized_infra(good_df)

    good_base = {
        "group": "GOOD_BASELINE",
        "num_bookings": len(good_df),

        "avg_booking_value": good_df["booking_value"].mean(),
        "avg_lead_time_days": good_df["lead_time_days"].mean(),

        "cancellation_rate_pct": good_df["is_cancelled"].mean() * 100,
        "avg_cancellation_delay_days": good_df.loc[good_df["is_cancelled"] == 1, "cancel_delay_days"].mean(),

        "dispute_rate_pct": good_df["is_disputed"].mean() * 100,
        "avg_chargeback_amount": good_df.loc[good_df["is_disputed"] == 1, "chargeback_amount"].mean(),
        "avg_loss_amount": good_df.loc[good_df["is_disputed"] == 1, "final_loss_amount"].mean(),

        "proxy_usage_rate_pct": good_df["proxy_flag"].mean() * 100,
        "international_booking_rate_pct": (good_df["route_type"] == "international").mean() * 100,
    }
    good_base.update(good_infra)
    good_base_df = pd.DataFrame([good_base])

    # Formatting
    round3 = [
        "avg_booking_value", "avg_lead_time_days", "avg_cancellation_delay_days",
        "avg_chargeback_amount", "avg_loss_amount",
        "avg_unique_ips_per_user", "avg_unique_devices_per_user",
        "avg_ip_churn_rate", "avg_device_churn_rate"
    ]
    round2 = [
        "cancellation_rate_pct", "dispute_rate_pct", "proxy_usage_rate_pct", "international_booking_rate_pct"
    ]

    for c in round3:
        if c in fraud_summary.columns:
            fraud_summary[c] = fraud_summary[c].round(3)
        if c in good_base_df.columns:
            good_base_df[c] = good_base_df[c].round(3)

    for c in round2:
        if c in fraud_summary.columns:
            fraud_summary[c] = fraud_summary[c].round(2)
        if c in good_base_df.columns:
            good_base_df[c] = good_base_df[c].round(2)

    # -----------------------------
    # Relevant columns per fraud reason
    # -----------------------------
    reason_cols = {
        "user has abnormally high cancellation rate": [
            "num_bookings", "cancellation_rate_pct", "avg_cancellation_delay_days",
            "avg_booking_value", "avg_unique_ips_per_user", "avg_ip_churn_rate"
        ],
        "credit bustout: high value international + high loss": [
            "num_bookings", "international_booking_rate_pct", "avg_booking_value",
            "dispute_rate_pct", "avg_chargeback_amount", "avg_loss_amount",
            "avg_unique_ips_per_user", "avg_ip_churn_rate"
        ],
        "new user with risky infra + abnormal velocity": [
            "num_bookings", "avg_lead_time_days", "proxy_usage_rate_pct",
            "avg_unique_ips_per_user", "avg_ip_churn_rate",
            "avg_unique_devices_per_user", "avg_device_churn_rate",
            "dispute_rate_pct"
        ],
        "new device/ip + short lead time": [
            "num_bookings", "avg_lead_time_days", "proxy_usage_rate_pct",
            "avg_unique_ips_per_user", "avg_ip_churn_rate",
            "avg_unique_devices_per_user", "avg_device_churn_rate",
            "avg_booking_value"
        ],
        "shared infra across multiple fraud users": [
            "num_bookings", "proxy_usage_rate_pct",
            "avg_unique_ips_per_user", "avg_ip_churn_rate",
            "avg_unique_devices_per_user", "avg_device_churn_rate"
        ],
        "burst/automation pattern in activity": [
            "num_bookings", "proxy_usage_rate_pct",
            "avg_unique_ips_per_user", "avg_ip_churn_rate",
            "avg_unique_devices_per_user", "avg_device_churn_rate",
            "avg_lead_time_days"
        ],
        "fraud user suspicious booking pattern": [
            "num_bookings", "avg_booking_value", "avg_lead_time_days",
            "proxy_usage_rate_pct", "dispute_rate_pct",
            "avg_unique_ips_per_user", "avg_ip_churn_rate"
        ]
    }

    default_cols = [
        "num_bookings", "avg_booking_value", "avg_lead_time_days",
        "proxy_usage_rate_pct", "dispute_rate_pct",
        "avg_unique_ips_per_user", "avg_ip_churn_rate"
    ]

    # -----------------------------
    # Print grouped comparison
    # -----------------------------
    for reason in fraud_summary["fraud_reason"].unique():
        cols = reason_cols.get(reason, default_cols)

        print("\n" + "="*120)
        print(f"FRAUD REASON: {reason.upper()}")
        print("="*120)

        fraud_row = fraud_summary[fraud_summary["fraud_reason"] == reason][["group"] + cols]
        print("\n[FRAUD]")
        print(fraud_row.to_string(index=False))

        good_row = good_base_df[["group"] + cols]
        print("\n[GOOD BASELINE]")
        print(good_row.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze fraud reasons vs good baseline.")
    parser.add_argument("data_dir", nargs="?", help="Path to Data/Tables directory.")
    args = parser.parse_args()
    
    analyze_booking_fraud(args.data_dir)
