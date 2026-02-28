#!/usr/bin/env python3
"""
Standalone sanity-check script for the B2B Travel Fraud Detection dataset.

Loads the generated CSVs from Data/Tables and runs every structural,
referential-integrity, and cross-table consistency check.

Usage:
    python cell8_sanity_checks.py              # run from Code/ directory
    python cell8_sanity_checks.py /path/to/Tables   # explicit data dir
"""

import os
import sys
import pandas as pd

# ============================================================
# CONFIGURATION / CONSTANTS
# ============================================================
N_BOOKINGS = 8000

DATACENTER_ASN_RANGES = set(range(10000, 15000)) | set(range(30000, 35000))

# ============================================================
# RESOLVE DATA DIRECTORY
# ============================================================
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    # Default: ../Data/Tables relative to this script
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "Data", "Tables")

DATA_DIR = os.path.abspath(DATA_DIR)
assert os.path.isdir(DATA_DIR), f"Data directory not found: {DATA_DIR}"

# ============================================================
# LOAD TABLES
# ============================================================
print(f"📂 Loading CSVs from {DATA_DIR} …")
agency_master       = pd.read_csv(os.path.join(DATA_DIR, "agency_master.csv"))
user_master         = pd.read_csv(os.path.join(DATA_DIR, "user_master.csv"))
booking_fact        = pd.read_csv(os.path.join(DATA_DIR, "booking_fact.csv"))
session_context     = pd.read_csv(os.path.join(DATA_DIR, "session_context.csv"))
passenger_details   = pd.read_csv(os.path.join(DATA_DIR, "passenger_details.csv"))
post_booking_events = pd.read_csv(os.path.join(DATA_DIR, "post_booking_events.csv"))
booking_label_table = pd.read_csv(os.path.join(DATA_DIR, "booking_label_table.csv"))
print("✅ All 7 tables loaded.\n")

# ============================================================
# SANITY CHECKS
# ============================================================
print("=" * 80)
print("  SANITY CHECKS")
print("=" * 80)

# ------------------------------------------------------------------
# SECTION A — BASIC STRUCTURAL CHECKS
# ------------------------------------------------------------------
print("\n--- A. Basic Structural Checks ---")

assert 'device_id' not in booking_fact.columns
assert 'ip_id' not in booking_fact.columns
print("✅ booking_fact has no device_id / ip_id")

assert len(session_context) == N_BOOKINGS
fp_os_check = session_context.groupby("device_fingerprint")["os"].nunique()
fp_br_check = session_context.groupby("device_fingerprint")["browser"].nunique()
assert (fp_os_check == 1).all(), "FAIL: same fingerprint has different OS!"
assert (fp_br_check == 1).all(), "FAIL: same fingerprint has different browser!"
print("✅ Fingerprint → OS/Browser consistency: PASSED")

assert (session_context['is_vpn_or_proxy'] == session_context['asn'].isin(DATACENTER_ASN_RANGES).astype(int)).all()
print("✅ VPN/Proxy = ASN-derived only: PASSED")

assert len(passenger_details) >= N_BOOKINGS
assert 'passenger_email_domain' in passenger_details.columns
assert 'is_suspicious_domain' in passenger_details.columns
assert 'email_domain_match_flag' not in passenger_details.columns
assert 'is_known_employee' not in passenger_details.columns
assert 'email_domain_match_flag' in user_master.columns
assert 'agency_email_domain' in agency_master.columns
assert 'device_switch_flag' in session_context.columns
print("✅ Column placement: PASSED")

# Legit users should never have device_switch_flag = 1 (they use 1 sticky fingerprint)
sc_with_user = session_context.merge(
    booking_fact[["booking_id", "user_id"]], on="booking_id"
).merge(
    user_master[["user_id", "user_fraud_label"]], on="user_id"
)
legit_device_switches = sc_with_user[
    (sc_with_user["user_fraud_label"] == 0) & (sc_with_user["device_switch_flag"] == 1)
]
assert len(legit_device_switches) == 0, (
    f"FAIL: {len(legit_device_switches)} legit-user sessions have device_switch_flag=1!"
)
print("✅ Legit users always use single device (device_switch_flag=0): PASSED")

# ------------------------------------------------------------------
# SECTION B — REFERENTIAL INTEGRITY (no orphan keys)
# ------------------------------------------------------------------
print("\n--- B. Referential Integrity (no orphan keys) ---")

bf_ids = set(booking_fact["booking_id"])
bl_ids = set(booking_label_table["booking_id"])
assert bf_ids == bl_ids, (
    f"FAIL: booking_fact ↔ booking_label_table mismatch! "
    f"Only in fact: {bf_ids - bl_ids}, Only in label: {bl_ids - bf_ids}"
)
print("✅ booking_fact ↔ booking_label_table: booking_id sets match exactly")

sc_ids = set(session_context["booking_id"])
assert sc_ids == bf_ids, (
    f"FAIL: session_context ↔ booking_fact mismatch! "
    f"Only in session: {sc_ids - bf_ids}, Only in fact: {bf_ids - sc_ids}"
)
print("✅ session_context ↔ booking_fact: booking_id sets match exactly")

pb_ids = set(post_booking_events["booking_id"])
assert pb_ids == bf_ids, (
    f"FAIL: post_booking_events ↔ booking_fact mismatch! "
    f"Only in post: {pb_ids - bf_ids}, Only in fact: {bf_ids - pb_ids}"
)
print("✅ post_booking_events ↔ booking_fact: booking_id sets match exactly")

pax_bids = set(passenger_details["booking_id"])
assert pax_bids.issubset(bf_ids), (
    f"FAIL: passenger_details has booking_ids not in booking_fact: {pax_bids - bf_ids}"
)
assert bf_ids.issubset(pax_bids), (
    f"FAIL: booking_fact has booking_ids with no passengers: {bf_ids - pax_bids}"
)
print("✅ passenger_details ↔ booking_fact: every booking has passengers, no orphans")

bf_uids = set(booking_fact["user_id"])
um_uids = set(user_master["user_id"])
assert bf_uids.issubset(um_uids), (
    f"FAIL: booking_fact has user_ids not in user_master: {bf_uids - um_uids}"
)
print("✅ booking_fact → user_master: all user_ids valid")

bf_aids = set(booking_fact["agency_id"])
am_aids = set(agency_master["agency_id"])
assert bf_aids.issubset(am_aids), (
    f"FAIL: booking_fact has agency_ids not in agency_master: {bf_aids - am_aids}"
)
print("✅ booking_fact → agency_master: all agency_ids valid")

um_aids = set(user_master["agency_id"])
assert um_aids.issubset(am_aids), (
    f"FAIL: user_master has agency_ids not in agency_master: {um_aids - am_aids}"
)
print("✅ user_master → agency_master: all agency_ids valid")

# ------------------------------------------------------------------
# SECTION C — CROSS-TABLE FRAUD LABEL CONSISTENCY
# ------------------------------------------------------------------
print("\n--- C. Cross-Table Fraud Label Consistency ---")

# C1: Every fraud booking must belong to a fraud user
fraud_bids = booking_label_table[booking_label_table["fraud_label"] == 1]["booking_id"].values
fraud_booking_users = booking_fact[booking_fact["booking_id"].isin(fraud_bids)]["user_id"].unique()
fraud_uids_from_um = set(user_master[user_master["user_fraud_label"] == 1]["user_id"])
bad_users = set(fraud_booking_users) - fraud_uids_from_um
assert len(bad_users) == 0, (
    f"FAIL: {len(bad_users)} fraud bookings belong to NON-fraud users: {bad_users}"
)
print(f"✅ C1: All {len(fraud_bids)} fraud bookings belong to fraud-labeled users")

# C2: Check if any good bookings belong to fraud users (by design this can happen)
good_bids = booking_label_table[booking_label_table["fraud_label"] == 0]["booking_id"].values
good_booking_users = booking_fact[booking_fact["booking_id"].isin(good_bids)]["user_id"].unique()
good_uids_from_um = set(user_master[user_master["user_fraud_label"] == 0]["user_id"])
good_bookings_by_fraud_users = set(good_booking_users) - good_uids_from_um
if len(good_bookings_by_fraud_users) > 0:
    good_by_fraud = booking_fact[
        (booking_fact["booking_id"].isin(good_bids)) &
        (booking_fact["user_id"].isin(good_bookings_by_fraud_users))
    ]
    print(f"⚠️  C2: {len(good_bookings_by_fraud_users)} fraud users also have "
          f"GOOD bookings (by design)")
    print(f"      → {len(good_by_fraud)} good bookings from fraud users "
          f"(out of {len(good_bids)} total good)")
else:
    print("✅ C2: All good bookings belong to good (legit) users")

# C3: Agency linkage consistency
bf_check = booking_fact[["booking_id", "user_id", "agency_id"]].merge(
    user_master[["user_id", "agency_id"]], on="user_id", suffixes=("_booking", "_user")
)
agency_mismatch = bf_check[bf_check["agency_id_booking"] != bf_check["agency_id_user"]]
assert len(agency_mismatch) == 0, (
    f"FAIL: {len(agency_mismatch)} bookings have agency_id mismatch between "
    f"booking_fact and user_master!\n{agency_mismatch.head(5).to_string()}"
)
print(f"✅ C3: Agency linkage consistent — booking_fact.agency_id matches "
      f"user_master.agency_id for all {len(bf_check)} bookings")

# C4: fraud_reason ↔ user_fraud_type alignment
reason_type_map = {
    "user has abnormally high cancellation rate": "cancellation_abuser",
    "credit bustout: high value international + high loss": "credit_bustout_user",
    "new user with risky infra + abnormal velocity": "new_synthetic_user",
    "new device/ip + short lead time": "account_takeover",
    "shared infra across multiple fraud users": "ring_operator",
    "burst/automation pattern in activity": "bot_booking",
}
fraud_labels_with_user = (
    booking_label_table[booking_label_table["fraud_label"] == 1]
    .merge(booking_fact[["booking_id", "user_id"]], on="booking_id")
    .merge(user_master[["user_id", "user_fraud_type"]], on="user_id")
)
mismatches = []
for _, row in fraud_labels_with_user.iterrows():
    expected_type = reason_type_map.get(row["fraud_reason"])
    if expected_type is not None and expected_type != row["user_fraud_type"]:
        mismatches.append({
            "booking_id": row["booking_id"],
            "fraud_reason": row["fraud_reason"],
            "expected_type": expected_type,
            "actual_type": row["user_fraud_type"],
        })
if len(mismatches) == 0:
    print(f"✅ C4: fraud_reason ↔ user_fraud_type aligned for all "
          f"{len(fraud_labels_with_user)} fraud bookings")
else:
    print(f"❌ C4 FAIL: {len(mismatches)} bookings have fraud_reason ↔ "
          f"user_fraud_type mismatch!")
    for m in mismatches[:5]:
        print(f"   {m}")
    assert False, "Cross-table fraud_reason ↔ user_fraud_type mismatch detected!"

# ------------------------------------------------------------------
# SECTION D — POST-BOOKING EVENTS CONSISTENCY WITH FRAUD TYPE
# ------------------------------------------------------------------
print("\n--- D. Post-Booking Events ↔ Fraud Type Consistency ---")

pbe_labeled = (
    post_booking_events
    .merge(booking_label_table[["booking_id", "fraud_label", "fraud_reason"]], on="booking_id")
    .merge(booking_fact[["booking_id", "user_id"]], on="booking_id")
    .merge(user_master[["user_id", "user_fraud_type"]], on="user_id")
)

# D1: cancellation_abuser fraud bookings should have higher cancel rate
cancel_abuse_fraud = pbe_labeled[
    (pbe_labeled["user_fraud_type"] == "cancellation_abuser") &
    (pbe_labeled["fraud_label"] == 1)
]
overall_cancel_rate = pbe_labeled["is_cancelled"].mean()
if len(cancel_abuse_fraud) > 0:
    ca_cancel_rate = cancel_abuse_fraud["is_cancelled"].mean()
    print(f"   D1: cancellation_abuser (fraud) cancel rate: "
          f"{ca_cancel_rate*100:.1f}% vs overall: {overall_cancel_rate*100:.1f}%")
    assert ca_cancel_rate > overall_cancel_rate, \
        "FAIL: cancellation_abuser fraud bookings should have higher cancel rate!"
    print("✅ D1: cancellation_abuser fraud bookings have elevated cancel rate")

# D2: credit_bustout_user fraud bookings should have higher dispute rate
bustout_fraud = pbe_labeled[
    (pbe_labeled["user_fraud_type"] == "credit_bustout_user") &
    (pbe_labeled["fraud_label"] == 1)
]
overall_dispute_rate = pbe_labeled["is_disputed"].mean()
if len(bustout_fraud) > 0:
    cb_dispute_rate = bustout_fraud["is_disputed"].mean()
    print(f"   D2: credit_bustout_user (fraud) dispute rate: "
          f"{cb_dispute_rate*100:.1f}% vs overall: {overall_dispute_rate*100:.1f}%")
    assert cb_dispute_rate > overall_dispute_rate, \
        "FAIL: credit_bustout_user fraud bookings should have higher dispute rate!"
    print("✅ D2: credit_bustout_user fraud bookings have elevated dispute rate")

# D3: Good bookings should have lower cancel + dispute rates than fraud bookings
fraud_pbe = pbe_labeled[pbe_labeled["fraud_label"] == 1]
good_pbe = pbe_labeled[pbe_labeled["fraud_label"] == 0]
fraud_cancel = fraud_pbe["is_cancelled"].mean()
good_cancel = good_pbe["is_cancelled"].mean()
fraud_dispute = fraud_pbe["is_disputed"].mean()
good_dispute = good_pbe["is_disputed"].mean()
print(f"   D3: Cancel rate  — Fraud: {fraud_cancel*100:.1f}% vs Good: {good_cancel*100:.1f}%")
print(f"   D3: Dispute rate — Fraud: {fraud_dispute*100:.1f}% vs Good: {good_dispute*100:.1f}%")

# ------------------------------------------------------------------
# SECTION E — SESSION CONTEXT ↔ FRAUD LABEL CONSISTENCY
# ------------------------------------------------------------------
print("\n--- E. Session Context ↔ Fraud Label Consistency ---")

sc_with_label = (
    session_context
    .merge(booking_label_table[["booking_id", "fraud_label"]], on="booking_id")
    .merge(booking_fact[["booking_id", "user_id"]], on="booking_id")
    .merge(user_master[["user_id", "user_fraud_type"]], on="user_id")
)

# E1: Fraud bookings should have higher VPN/proxy rate
fraud_vpn = sc_with_label[sc_with_label["fraud_label"] == 1]["is_vpn_or_proxy"].mean()
good_vpn = sc_with_label[sc_with_label["fraud_label"] == 0]["is_vpn_or_proxy"].mean()
print(f"   E1: VPN/Proxy rate — Fraud: {fraud_vpn*100:.1f}% vs Good: {good_vpn*100:.1f}%")
assert fraud_vpn > good_vpn, "FAIL: Fraud bookings should have higher VPN/proxy rate!"
print("✅ E1: Fraud bookings have elevated VPN/proxy rate")

# E2: Avg unique IPs per device — fraud should be higher (IP masking signal)
fraud_ip_per_fp = (
    sc_with_label[sc_with_label['fraud_label'] == 1]
    .groupby('device_fingerprint')['ip_address'].nunique().mean()
)
legit_ip_per_fp = (
    sc_with_label[sc_with_label['fraud_label'] == 0]
    .groupby('device_fingerprint')['ip_address'].nunique().mean()
)
print(f"   E2: Avg unique IPs per device — Fraud: {fraud_ip_per_fp:.2f}, "
      f"Good: {legit_ip_per_fp:.2f}")
assert fraud_ip_per_fp > legit_ip_per_fp, \
    "FAIL: Fraud should have more IPs per device (IP masking)!"
print("✅ E2: Fraud has higher IP diversity per device (IP masking detected)")

# E3: Booking time — fraud should be faster (bots/scripts)
fraud_ttb = sc_with_label[sc_with_label['fraud_label'] == 1]['time_to_book_seconds'].mean()
good_ttb = sc_with_label[sc_with_label['fraud_label'] == 0]['time_to_book_seconds'].mean()
print(f"   E3: Avg booking time — Fraud: {fraud_ttb:.0f}s vs Good: {good_ttb:.0f}s")
assert fraud_ttb < good_ttb, "FAIL: Fraud bookings should be faster (bot-like)!"
print("✅ E3: Fraud bookings have lower booking time (bot/script pattern detected)")

# E4: Device-switch rate — fraud should be higher (multi-device usage)
fraud_dev_switch = sc_with_label[sc_with_label["fraud_label"] == 1]["device_switch_flag"].mean()
good_dev_switch = sc_with_label[sc_with_label["fraud_label"] == 0]["device_switch_flag"].mean()
print(f"   E4: Device-switch rate — Fraud: {fraud_dev_switch*100:.1f}% vs Good: {good_dev_switch*100:.1f}%")
assert fraud_dev_switch > good_dev_switch, \
    "FAIL: Fraud bookings should have higher device-switch rate!"
print("✅ E4: Fraud bookings have elevated device-switch rate")

# ------------------------------------------------------------------
# SECTION F — PASSENGER DETAILS ↔ FRAUD LABEL CONSISTENCY
# ------------------------------------------------------------------
print("\n--- F. Passenger Details ↔ Fraud Label Consistency ---")

pax_labeled = (
    passenger_details
    .merge(booking_label_table[["booking_id", "fraud_label"]], on="booking_id")
    .merge(booking_fact[["booking_id", "agency_id"]], on="booking_id")
    .merge(agency_master[["agency_id", "agency_email_domain"]], on="agency_id")
)

# F1: Corporate domain match rate
pax_labeled["domain_matches_agency"] = (
    pax_labeled["passenger_email_domain"] == pax_labeled["agency_email_domain"]
).astype(int)
fraud_domain_match = pax_labeled[pax_labeled["fraud_label"] == 1]["domain_matches_agency"].mean()
good_domain_match = pax_labeled[pax_labeled["fraud_label"] == 0]["domain_matches_agency"].mean()
print(f"   F1: Passenger email matches agency domain — "
      f"Fraud: {fraud_domain_match*100:.1f}% vs Good: {good_domain_match*100:.1f}%")
assert good_domain_match > fraud_domain_match, \
    "FAIL: Good bookings should have higher agency domain match rate!"
print("✅ F1: Good bookings have higher agency-email domain match")

# F2: Suspicious (throwaway) email domain usage — uses pre-computed is_suspicious_domain
fraud_suspicious = pax_labeled[pax_labeled["fraud_label"] == 1]["is_suspicious_domain"].mean()
good_suspicious = pax_labeled[pax_labeled["fraud_label"] == 0]["is_suspicious_domain"].mean()
print(f"   F2: Suspicious domain usage — "
      f"Fraud: {fraud_suspicious*100:.1f}% vs Good: {good_suspicious*100:.1f}%")
assert fraud_suspicious > good_suspicious, \
    "FAIL: Fraud bookings should have higher suspicious domain usage!"
print("✅ F2: Fraud bookings have higher suspicious (throwaway) domain usage")

# ------------------------------------------------------------------
# SECTION G — SUMMARY STATS
# ------------------------------------------------------------------
print("\n--- G. Summary Statistics ---")
print(f"  Tables generated: 7")
print(f"  agency_master:       {len(agency_master):>6} rows")
print(f"  user_master:         {len(user_master):>6} rows  "
      f"(fraud: {user_master['user_fraud_label'].sum()})")
print(f"  booking_fact:        {len(booking_fact):>6} rows")
print(f"  session_context:     {len(session_context):>6} rows")
print(f"  passenger_details:   {len(passenger_details):>6} rows")
print(f"  post_booking_events: {len(post_booking_events):>6} rows")
print(f"  booking_label_table: {len(booking_label_table):>6} rows  "
      f"(fraud: {booking_label_table['fraud_label'].sum()})")
print(f"\n{'='*80}")
print("  ✅ ALL CROSS-TABLE FRAUD LABEL CONSISTENCY CHECKS PASSED")
print(f"{'='*80}")
