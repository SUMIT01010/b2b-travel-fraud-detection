import pandas as pd
import numpy as np
import os
import argparse

def analyze_user_fraud(data_dir=None):
    """
    Analyzes and compares fraud users vs good users based on the generated synthetic dataset.
    
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
    # Merge booking-level view
    # -----------------------------
    df = (book.merge(label, on="booking_id", how="left")
              .merge(events, on="booking_id", how="left")
              .merge(users[["user_id", "user_fraud_label", "user_fraud_type",
                           "user_age_days", "avg_logins_per_day", "failed_login_ratio"]],
                     on="user_id", how="left")
              .merge(sess[["booking_id", "ip_address", "device_fingerprint", "is_vpn_or_proxy"]], 
                     on="booking_id", how="left"))

    df["proxy_flag"] = df["is_vpn_or_proxy"].fillna(0).astype(int)

    # -----------------------------
    # User-level metrics table
    # -----------------------------
    user_metrics = (
        df.groupby("user_id")
          .agg(
              user_fraud_label=("user_fraud_label", "max"),
              user_fraud_type=("user_fraud_type", "max"),
              user_age_days=("user_age_days", "max"),
              avg_logins=("avg_logins_per_day", "max"),
              failed_login_ratio=("failed_login_ratio", "max"),

              total_bookings=("booking_id", "count"),
              fraud_bookings=("fraud_label", "sum"),

              cancels=("is_cancelled", "sum"),
              disputes=("is_disputed", "sum"),

              total_loss=("final_loss_amount", "sum"),
              avg_booking_value=("booking_value", "mean"),
              max_booking_value=("booking_value", "max"),

              proxy_rate=("proxy_flag", "mean"),
              unique_ips=("ip_address", "nunique"),
          )
          .reset_index()
    )

    user_metrics["cancel_rate"] = (user_metrics["cancels"] / user_metrics["total_bookings"]).round(3)
    user_metrics["dispute_rate"] = (user_metrics["disputes"] / user_metrics["total_bookings"]).round(3)
    user_metrics["proxy_rate_pct"] = (user_metrics["proxy_rate"] * 100).round(2)

    fraud_users = user_metrics[user_metrics["user_fraud_label"] == 1].copy()
    good_users  = user_metrics[user_metrics["user_fraud_label"] == 0].copy()

    print(f"📂 Analyzing data from: {data_dir}")
    print(f"✅ Fraud users: {len(fraud_users)}")
    print(f"✅ Good users : {len(good_users)}")

    # -----------------------------
    # Relevant columns per fraud type
    # -----------------------------
    cols_map = {
        "cancellation_abuser": [
            "user_id", "total_bookings", "cancels", "cancel_rate"
        ],
        "account_takeover": [
            "user_id", "failed_login_ratio", "avg_logins", "unique_ips",
            "proxy_rate_pct", "fraud_bookings"
        ],
        "new_synthetic_user": [
            "user_id", "user_age_days", "total_bookings", "fraud_bookings",
            "proxy_rate_pct", "unique_ips"
        ],
        "credit_bustout_user": [
            "user_id", "fraud_bookings", "max_booking_value",
            "avg_booking_value", "disputes", "total_loss"
        ],
        "bot_booking": [
            "user_id", "avg_logins", "total_bookings", "fraud_bookings", "unique_ips"
        ],
        "ring_operator": [
            "user_id", "fraud_bookings", "unique_ips", "proxy_rate_pct", "total_bookings"
        ],
    }

    # -----------------------------
    # Comparison loop
    # -----------------------------
    def print_user_type_compare(ftype, top_n=8):
        fraud_sub = fraud_users[fraud_users["user_fraud_type"] == ftype].copy()
        if fraud_sub.empty:
            print(f"\n❌ No fraud users found for: {ftype}")
            return

        # Sort per fraud type
        if ftype == "cancellation_abuser":
            fraud_sub = fraud_sub.sort_values(["cancel_rate", "cancels"], ascending=False)
        elif ftype == "credit_bustout_user":
            fraud_sub = fraud_sub.sort_values(["total_loss", "max_booking_value"], ascending=False)
        elif ftype == "account_takeover":
            fraud_sub = fraud_sub.sort_values(["failed_login_ratio", "unique_ips"], ascending=False)
        elif ftype == "new_synthetic_user":
            fraud_sub = fraud_sub.sort_values(["user_age_days", "total_bookings"], ascending=[True, False])
        elif ftype == "bot_booking":
            fraud_sub = fraud_sub.sort_values(["avg_logins", "total_bookings"], ascending=False)
        elif ftype == "ring_operator":
            fraud_sub = fraud_sub.sort_values(["unique_ips", "proxy_rate_pct"], ascending=False)

        cols = cols_map[ftype]

        print("\n" + "="*110)
        print(f"USER FRAUD TYPE: {ftype.upper()}")
        print("="*110)

        print("\n[FRAUD USERS - SAMPLE]")
        print(fraud_sub[cols].head(top_n).to_string(index=False))

        # GOOD baseline = average of those same metrics (excluding user_id)
        metric_cols = [c for c in cols if c != "user_id"]
        good_avg = good_users[metric_cols].mean(numeric_only=True).to_frame().T
        good_avg.insert(0, "baseline_group", "GOOD_USERS_AVG")

        # format small decimals
        for c in good_avg.columns:
            if c in ["failed_login_ratio", "cancel_rate", "dispute_rate"]:
                good_avg[c] = good_avg[c].apply(lambda x: round(x, 4))
            elif c.endswith("_pct"):
                good_avg[c] = good_avg[c].apply(lambda x: round(x, 2))
            elif c == "baseline_group":
                pass
            else:
                good_avg[c] = good_avg[c].apply(lambda x: round(x, 3))

        print("\n[GOOD USERS BASELINE - AVERAGE]")
        print(good_avg.to_string(index=False))

    for f_type in cols_map.keys():
        print_user_type_compare(f_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze fraud user types vs good users.")
    parser.add_argument("data_dir", nargs="?", help="Path to Data/Tables directory.")
    args = parser.parse_args()
    
    analyze_user_fraud(args.data_dir)
