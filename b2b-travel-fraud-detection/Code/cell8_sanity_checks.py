# ============================================================
# CELL 8 — SAVE FILES & SANITY CHECKS
# ============================================================
agency_master.to_csv(os.path.join(OUT_DIR, "agency_master.csv"), index=False)
user_master.to_csv(os.path.join(OUT_DIR, "user_master.csv"), index=False)
booking_fact.to_csv(os.path.join(OUT_DIR, "booking_fact.csv"), index=False)
session_context.to_csv(os.path.join(OUT_DIR, "session_context.csv"), index=False)
passenger_details.to_csv(os.path.join(OUT_DIR, "passenger_details.csv"), index=False)
post_booking_events.to_csv(os.path.join(OUT_DIR, "post_booking_events.csv"), index=False)
booking_label_table.to_csv(os.path.join(OUT_DIR, "booking_label_table.csv"), index=False)

for old_file in ["device_master.csv", "ip_master.csv"]:
    old_path = os.path.join(OUT_DIR, old_file)
    if os.path.exists(old_path):
        os.remove(old_path)
        print(f"🗑️  Removed old {old_file}")

with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(OUT_DIR):
        z.write(os.path.join(OUT_DIR, f), arcname=f)

print("\n✅ Dataset generated!")
print("Folder:", OUT_DIR)
print("Zip:", ZIP_NAME)

print("\n" + "="*80)
print("  SANITY CHECKS")
print("="*80)

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
assert 'email_domain_match_flag' not in passenger_details.columns
assert 'is_known_employee' not in passenger_details.columns
assert 'email_domain_match_flag' in user_master.columns
assert 'agency_email_domain' in agency_master.columns
print("✅ Column placement: PASSED")

# ------------------------------------------------------------------
# SECTION B — REFERENTIAL INTEGRITY (no orphan keys)
# ------------------------------------------------------------------
print("\n--- B. Referential Integrity (no orphan keys) ---")

# Every booking_id in booking_fact exists in booking_label_table and vice-versa
bf_ids = set(booking_fact["booking_id"])
bl_ids = set(booking_label_table["booking_id"])
assert bf_ids == bl_ids, f"FAIL: booking_fact ↔ booking_label_table mismatch! Only in fact: {bf_ids - bl_ids}, Only in label: {bl_ids - bf_ids}"
print("✅ booking_fact ↔ booking_label_table: booking_id sets match exactly")

# Every booking_id in session_context exists in booking_fact
sc_ids = set(session_context["booking_id"])
assert sc_ids == bf_ids, f"FAIL: session_context ↔ booking_fact mismatch! Only in session: {sc_ids - bf_ids}, Only in fact: {bf_ids - sc_ids}"
print("✅ session_context ↔ booking_fact: booking_id sets match exactly")

# Every booking_id in post_booking_events exists in booking_fact
pb_ids = set(post_booking_events["booking_id"])
assert pb_ids == bf_ids, f"FAIL: post_booking_events ↔ booking_fact mismatch! Only in post: {pb_ids - bf_ids}, Only in fact: {bf_ids - pb_ids}"
print("✅ post_booking_events ↔ booking_fact: booking_id sets match exactly")

# Every booking_id in passenger_details exists in booking_fact
pax_bids = set(passenger_details["booking_id"])
assert pax_bids.issubset(bf_ids), f"FAIL: passenger_details has booking_ids not in booking_fact: {pax_bids - bf_ids}"
assert bf_ids.issubset(pax_bids), f"FAIL: booking_fact has booking_ids with no passengers: {bf_ids - pax_bids}"
print("✅ passenger_details ↔ booking_fact: every booking has passengers, no orphans")

# Every user_id in booking_fact exists in user_master
bf_uids = set(booking_fact["user_id"])
um_uids = set(user_master["user_id"])
assert bf_uids.issubset(um_uids), f"FAIL: booking_fact has user_ids not in user_master: {bf_uids - um_uids}"
print("✅ booking_fact → user_master: all user_ids valid")

# Every agency_id in booking_fact exists in agency_master
bf_aids = set(booking_fact["agency_id"])
am_aids = set(agency_master["agency_id"])
assert bf_aids.issubset(am_aids), f"FAIL: booking_fact has agency_ids not in agency_master: {bf_aids - am_aids}"
print("✅ booking_fact → agency_master: all agency_ids valid")

# Every agency_id in user_master exists in agency_master
um_aids = set(user_master["agency_id"])
assert um_aids.issubset(am_aids), f"FAIL: user_master has agency_ids not in agency_master: {um_aids - am_aids}"
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
assert len(bad_users) == 0, f"FAIL: {len(bad_users)} fraud bookings belong to NON-fraud users: {bad_users}"
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
    print(f"⚠️  C2: {len(good_bookings_by_fraud_users)} fraud users also have GOOD bookings (by design)")
    print(f"      → {len(good_by_fraud)} good bookings from fraud users (out of {len(good_bids)} total good)")
else:
    print("✅ C2: All good bookings belong to good (legit) users")

# C3: Agency linkage consistency — booking_fact.agency_id must match
#     the agency_id of the user in user_master for that booking
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

# C4: fraud_reason in booking_label_table aligns with user_fraud_type in user_master
reason_type_map = {
    "user has abnormally high cancellation rate": "cancellation_abuser",
    "credit bustout: high value international + high loss": "credit_bustout_user",
    "new user with risky infra + abnormal velocity": "new_synthetic_user",
    "new device/ip + short lead time": "account_takeover",
    "shared infra across multiple fraud users": "ring_operator",
    "burst/automation pattern in activity": "bot_booking",
}
fraud_labels_with_user = booking_label_table[booking_label_table["fraud_label"] == 1].merge(
    booking_fact[["booking_id", "user_id"]], on="booking_id"
).merge(
    user_master[["user_id", "user_fraud_type"]], on="user_id"
)
mismatches = []
for _, row in fraud_labels_with_user.iterrows():
    expected_type = reason_type_map.get(row["fraud_reason"])
    if expected_type is not None and expected_type != row["user_fraud_type"]:
        mismatches.append({
            "booking_id": row["booking_id"],
            "fraud_reason": row["fraud_reason"],
            "expected_type": expected_type,
            "actual_type": row["user_fraud_type"]
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

pbe_labeled = post_booking_events.merge(
    booking_label_table[["booking_id", "fraud_label", "fraud_reason"]], on="booking_id"
).merge(
    booking_fact[["booking_id", "user_id"]], on="booking_id"
).merge(
    user_master[["user_id", "user_fraud_type"]], on="user_id"
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
    print(f"✅ D1: cancellation_abuser fraud bookings have elevated cancel rate")

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
    print(f"✅ D2: credit_bustout_user fraud bookings have elevated dispute rate")

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

sc_with_label = session_context.merge(
    booking_label_table[["booking_id", "fraud_label"]], on="booking_id"
).merge(
    booking_fact[["booking_id", "user_id"]], on="booking_id"
).merge(
    user_master[["user_id", "user_fraud_type"]], on="user_id"
)

# E1: Fraud bookings should have higher VPN/proxy rate
fraud_vpn = sc_with_label[sc_with_label["fraud_label"] == 1]["is_vpn_or_proxy"].mean()
good_vpn = sc_with_label[sc_with_label["fraud_label"] == 0]["is_vpn_or_proxy"].mean()
print(f"   E1: VPN/Proxy rate — Fraud: {fraud_vpn*100:.1f}% vs Good: {good_vpn*100:.1f}%")
assert fraud_vpn > good_vpn, "FAIL: Fraud bookings should have higher VPN/proxy rate!"
print("✅ E1: Fraud bookings have elevated VPN/proxy rate")

# E2: Avg unique IPs per device — fraud should be higher (IP masking signal)
fraud_ip_per_fp = sc_with_label[sc_with_label['fraud_label']==1] \
    .groupby('device_fingerprint')['ip_address'].nunique().mean()
legit_ip_per_fp = sc_with_label[sc_with_label['fraud_label']==0] \
    .groupby('device_fingerprint')['ip_address'].nunique().mean()
print(f"   E2: Avg unique IPs per device — Fraud: {fraud_ip_per_fp:.2f}, Good: {legit_ip_per_fp:.2f}")
assert fraud_ip_per_fp > legit_ip_per_fp, \
    "FAIL: Fraud should have more IPs per device (IP masking)!"
print("✅ E2: Fraud has higher IP diversity per device (IP masking detected)")

# ------------------------------------------------------------------
# SECTION F — PASSENGER DETAILS ↔ FRAUD LABEL CONSISTENCY
# ------------------------------------------------------------------
print("\n--- F. Passenger Details ↔ Fraud Label Consistency ---")

pax_labeled = passenger_details.merge(
    booking_label_table[["booking_id", "fraud_label"]], on="booking_id"
).merge(
    booking_fact[["booking_id", "agency_id"]], on="booking_id"
).merge(
    agency_master[["agency_id", "agency_email_domain"]], on="agency_id"
)

# F1: Corporate domain match rate — legit passengers should match agency domain more
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

# F2: Throwaway email usage — fraud passengers should use more throwaway domains
throwaway_set = set(THROWAWAY_DOMAINS)
pax_labeled["is_throwaway"] = pax_labeled["passenger_email_domain"].isin(throwaway_set).astype(int)
fraud_throwaway = pax_labeled[pax_labeled["fraud_label"] == 1]["is_throwaway"].mean()
good_throwaway = pax_labeled[pax_labeled["fraud_label"] == 0]["is_throwaway"].mean()
print(f"   F2: Throwaway email usage — "
      f"Fraud: {fraud_throwaway*100:.1f}% vs Good: {good_throwaway*100:.1f}%")
assert fraud_throwaway > good_throwaway, \
    "FAIL: Fraud bookings should have higher throwaway email usage!"
print("✅ F2: Fraud bookings have higher throwaway email usage")

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
