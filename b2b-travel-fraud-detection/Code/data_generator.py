import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import uuid
import hashlib

def data_generator(
    n_bookings=8000,
    n_users=800,
    n_fraud_users=40,
    target_fraud_bookings=720,
    n_agencies=200,
    start_date=datetime(2025, 1, 1),
    out_dir=None,
    seed=42
):
    """
    Generates synthetic B2B travel fraud detection data.
    
    Args:
        n_bookings (int): Total number of bookings.
        n_users (int): Total number of unique users.
        n_fraud_users (int): Number of users tagged as fraudulent.
        target_fraud_bookings (int): Targeted number of fraudulent bookings.
        n_agencies (int): Number of travel agencies.
        start_date (datetime): Start date for bookings.
        out_dir (str, optional): Directory to save CSV files.
        seed (int): Random seed for reproducibility.
        
    Returns:
        dict: A dictionary containing generated DataFrames.
    """
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # --- Helpers -----------------------------------------------
    def make_uuid():
        return str(uuid.uuid4())

    def make_fingerprint(seed_str):
        """Deterministic hardware/cookie hash from a seed string."""
        return hashlib.sha256(seed_str.encode()).hexdigest()[:16]

    def random_ip():
        return ".".join(map(str, np.random.randint(1, 255, size=4)))

    # Datacenter ASN ranges — the ONLY way we detect VPN/proxy.
    DATACENTER_ASN_RANGES = set(range(10000, 15000)) | set(range(30000, 35000))

    # --- Email domain pools ------------------------------------
    CORPORATE_DOMAIN_POOL = [
        "travelcorp.com", "globeways.in", "flyhigh.ae", "skybound.co",
        "transwings.com", "jetsetgo.in", "voyagepro.sg", "airdesk.uk",
        "tripstation.com", "bookmyflight.in", "travease.ae", "routemaster.co",
        "flydeals.com", "wingspan.sg", "gotravel.in", "aerobiz.uk",
        "traveledge.com", "swiftfly.in", "jetstream.ae", "horizonair.co",
        "quicktrips.com", "flynext.sg", "tourhub.in", "skypaths.uk",
        "travelprime.com", "flybright.ae", "airsync.in", "globalwings.co",
        "travelworks.sg", "flyweb.com", "tripcraft.in", "aerolink.uk",
        "airventure.com", "flyscope.ae", "travelzen.in", "jetforce.co",
        "airnova.sg", "travelmax.com", "skyroute.in", "wingcraft.uk"
    ]

    GENERIC_DOMAINS = [
        "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
        "protonmail.com", "mail.com", "yandex.com", "aol.com",
        "zoho.com", "icloud.com"
    ]

    THROWAWAY_DOMAINS = [
        "tempmail.xyz", "fakeinbox.org", "trashmail.net", "guerrillamail.com",
        "mailinator.com", "throwaway.email", "discard.email", "sharklasers.com"
    ]

    # ============================================================
    # AGENCY MASTER
    # ============================================================
    agency_ids = [f"A{str(i).zfill(4)}" for i in range(1, n_agencies + 1)]
    agency_domains = [CORPORATE_DOMAIN_POOL[i % len(CORPORATE_DOMAIN_POOL)] for i in range(n_agencies)]

    agency_master = pd.DataFrame({
        "agency_id": agency_ids,
        "country": np.random.choice(["IN", "AE", "SG", "UK", "US"], size=n_agencies,
                                    p=[0.45, 0.15, 0.10, 0.15, 0.15]),
        "agency_age_days": np.random.randint(30, 3000, size=n_agencies),
        "kyc_status": np.random.choice(["verified", "pending", "failed"], size=n_agencies,
                                       p=[0.85, 0.12, 0.03]),
        "credit_limit": np.random.choice([5000, 10000, 25000, 50000, 100000, 200000, 500000],
                                         size=n_agencies,
                                         p=[0.20, 0.20, 0.20, 0.15, 0.15, 0.07, 0.03]),
        "status": np.random.choice(["active", "suspended"], size=n_agencies, p=[0.97, 0.03]),
        "agency_email_domain": agency_domains
    })

    agency_to_domain = dict(zip(agency_master["agency_id"], agency_master["agency_email_domain"]))

    # ============================================================
    # USER MASTER
    # ============================================================
    user_ids = [f"U{str(i).zfill(5)}" for i in range(1, n_users + 1)]
    user_agencies = np.random.choice(agency_ids, size=n_users)

    fraud_user_ids = np.random.choice(user_ids, size=n_fraud_users, replace=False)
    fraud_user_set = set(fraud_user_ids)

    fraud_type_counts = {
        "account_takeover": 7, "ring_operator": 7, "bot_booking": 7,
        "cancellation_abuser": 6, "credit_bustout_user": 7, "new_synthetic_user": 6
    }
    # Adjust fraud counts if n_fraud_users is different from default 40
    if n_fraud_users != 40:
        actual_sum = sum(fraud_type_counts.values())
        keys = list(fraud_type_counts.keys())
        for i in range(abs(n_fraud_users - actual_sum)):
            key = keys[i % len(keys)]
            fraud_type_counts[key] += 1 if n_fraud_users > actual_sum else -1

    balanced_fraud_types = []
    for k, v in fraud_type_counts.items():
        balanced_fraud_types.extend([k] * v)
    random.shuffle(balanced_fraud_types)
    fraud_type_map = dict(zip(fraud_user_ids, balanced_fraud_types))

    roles = np.random.choice(["agent", "admin", "finance"], size=n_users, p=[0.78, 0.12, 0.10])
    user_age_days = np.random.randint(30, 2500, size=n_users)
    avg_logins = np.clip(np.random.normal(6, 3.5, size=n_users), 0, 60)
    failed_login_ratio = np.clip(np.random.beta(1.5, 40, size=n_users), 0, 1)
    account_status = np.random.choice(["active", "locked"], size=n_users, p=[0.985, 0.015])
    email_domain_match = np.zeros(n_users, dtype=int)

    for i, uid in enumerate(user_ids):
        if uid in fraud_user_set:
            ftype = fraud_type_map[uid]
            if ftype == "account_takeover":
                failed_login_ratio[i] = np.random.uniform(0.18, 0.55)
                avg_logins[i] = np.random.uniform(10, 25)
            elif ftype == "bot_booking":
                failed_login_ratio[i] = np.random.uniform(0.05, 0.20)
                avg_logins[i] = np.random.uniform(25, 60)
            elif ftype == "ring_operator":
                failed_login_ratio[i] = np.random.uniform(0.03, 0.14)
                avg_logins[i] = np.random.uniform(12, 28)
            elif ftype == "cancellation_abuser":
                failed_login_ratio[i] = np.random.uniform(0.01, 0.10)
                avg_logins[i] = np.random.uniform(8, 18)
            elif ftype == "credit_bustout_user":
                failed_login_ratio[i] = np.random.uniform(0.01, 0.08)
                avg_logins[i] = np.random.uniform(10, 22)
            elif ftype == "new_synthetic_user":
                user_age_days[i] = np.random.randint(3, 40)
                failed_login_ratio[i] = np.random.uniform(0.02, 0.18)
                avg_logins[i] = np.random.uniform(12, 40)
            if np.random.rand() < 0.05:
                account_status[i] = "locked"

            if ftype in ["new_synthetic_user", "ring_operator"]:
                email_domain_match[i] = int(np.random.rand() < 0.15)
            elif ftype in ["bot_booking", "account_takeover"]:
                email_domain_match[i] = int(np.random.rand() < 0.35)
            elif ftype == "credit_bustout_user":
                email_domain_match[i] = int(np.random.rand() < 0.50)
            else:
                email_domain_match[i] = int(np.random.rand() < 0.75)
        else:
            email_domain_match[i] = int(np.random.rand() < 0.90)

    user_master = pd.DataFrame({
        "user_id": user_ids, "agency_id": user_agencies, "role": roles,
        "user_age_days": user_age_days,
        "avg_logins_per_day": np.round(avg_logins, 2),
        "failed_login_ratio": np.round(failed_login_ratio, 4),
        "account_status": account_status,
        "email_domain_match_flag": email_domain_match,
        "user_fraud_label": [1 if u in fraud_user_set else 0 for u in user_ids],
        "user_fraud_type": [fraud_type_map[u] if u in fraud_user_set else "legit" for u in user_ids],
    })

    user_to_agency = dict(zip(user_master["user_id"], user_master["agency_id"]))

    # ============================================================
    # BOOKING FACT
    # ============================================================
    booking_ids = [f"B{str(i).zfill(6)}" for i in range(1, n_bookings + 1)]
    booking_ts = [start_date + timedelta(minutes=int(x)) for x in np.random.randint(0, 365 * 24 * 60, size=n_bookings)]

    product_type = np.random.choice(["flight", "hotel", "package"], size=n_bookings, p=[0.6, 0.3, 0.1])
    route_type = np.random.choice(["domestic", "international"], size=n_bookings, p=[0.65, 0.35])
    origin_country = np.random.choice(["IN", "AE", "SG", "UK", "US"], size=n_bookings,
                                      p=[0.45, 0.15, 0.10, 0.15, 0.15])
    dest_country = np.where(route_type == "domestic", origin_country,
                            np.random.choice(["IN", "AE", "SG", "UK", "US", "TH", "FR", "DE"], size=n_bookings))
    lead_time_days = np.where(route_type == "domestic",
                              np.random.randint(1, 20, size=n_bookings),
                              np.random.randint(2, 90, size=n_bookings))

    def gen_value(rt):
        if rt == "domestic":
            return float(np.clip(np.random.lognormal(mean=5.3, sigma=0.55), 40, 12000))
        return float(np.clip(np.random.lognormal(mean=7.2, sigma=0.65), 200, 25000))

    booking_value = np.array([gen_value(rt) for rt in route_type])
    passengers_count = np.random.choice([1,2,3,4,5,6,7,8,9,10], size=n_bookings,
                                        p=[0.35,0.25,0.15,0.08,0.05,0.04,0.03,0.025,0.015,0.01])
    payment_method = np.random.choice(["credit_line", "card", "bank"], size=n_bookings, p=[0.75, 0.18, 0.07])
    booking_status = np.random.choice(["confirmed", "pending", "failed"], size=n_bookings, p=[0.92, 0.06, 0.02])

    fraud_booking_indices = np.random.choice(np.arange(n_bookings), size=target_fraud_bookings, replace=False)
    fraud_booking_set = set(fraud_booking_indices)

    fraud_users_list = list(fraud_user_ids)
    # Ensure fraud_users_for_bookings is not larger than fraud_users_list
    n_fraud_pool = min(32, len(fraud_users_list))
    fraud_users_for_bookings = np.random.choice(fraud_users_list, size=n_fraud_pool, replace=False)
    legit_users_list = [u for u in user_ids if u not in fraud_user_set]

    user_id_col = np.empty(n_bookings, dtype=object)
    agency_id_col = np.empty(n_bookings, dtype=object)

    for i in range(n_bookings):
        if i in fraud_booking_set:
            u = np.random.choice(fraud_users_for_bookings)
        else:
            u = np.random.choice(legit_users_list)
        user_id_col[i] = u
        agency_id_col[i] = user_to_agency[u]

    booking_fact = pd.DataFrame({
        "booking_id": booking_ids, "booking_ts": booking_ts,
        "agency_id": agency_id_col, "user_id": user_id_col,
        "product_type": product_type, "route_type": route_type,
        "origin_country": origin_country, "dest_country": dest_country,
        "lead_time_days": lead_time_days.astype(int),
        "booking_value": np.round(booking_value, 2),
        "passengers_count": passengers_count.astype(int),
        "payment_method": payment_method, "booking_status": booking_status
    })

    # ============================================================
    # SESSION CONTEXT
    # ============================================================
    N_FINGERPRINTS = 500
    N_IPS = 600

    all_fingerprints = [make_fingerprint(f"dev_{i}") for i in range(N_FINGERPRINTS)]
    all_ips = [random_ip() for _ in range(N_IPS)]

    fraud_fp_pool = list(np.random.choice(all_fingerprints, size=20, replace=False))
    # normal_fp_pool = [f for f in all_fingerprints if f not in set(fraud_fp_pool)]

    # fraud_ip_pool = list(np.random.choice(all_ips, size=24, replace=False))
    # normal_ip_pool = [ip for ip in all_ips if ip not in set(fraud_ip_pool)]
    
    # Redefine pools properly to avoid sets being slow in large lists if needed
    fraud_fp_set = set(fraud_fp_pool)
    normal_fp_pool = [f for f in all_fingerprints if f not in fraud_fp_set]
    
    fraud_ip_pool = list(np.random.choice(all_ips, size=24, replace=False))
    fraud_ip_set = set(fraud_ip_pool)
    normal_ip_pool = [ip for ip in all_ips if ip not in fraud_ip_set]

    os_choices = ["Windows", "macOS", "Android", "iOS"]
    os_probs = [0.45, 0.20, 0.25, 0.10]
    browser_choices = ["Chrome", "Safari", "Edge", "Firefox"]
    browser_probs = [0.65, 0.12, 0.15, 0.08]

    fp_to_os = {fp: np.random.choice(os_choices, p=os_probs) for fp in all_fingerprints}
    fp_to_browser = {fp: np.random.choice(browser_choices, p=browser_probs) for fp in all_fingerprints}

    user_sticky_fps = {}
    user_sticky_ips = {}

    for u in legit_users_list:
        user_sticky_fps[u] = [np.random.choice(normal_fp_pool)]
        k_ips = np.random.choice([1, 2], p=[0.85, 0.15])
        user_sticky_ips[u] = list(np.random.choice(normal_ip_pool, size=k_ips, replace=False))

    for u in fraud_users_list:
        utype = user_master.loc[user_master["user_id"] == u, "user_fraud_type"].values[0]
        if utype in ["ring_operator", "bot_booking"]:
            k_fp = np.random.choice([2, 3, 4], p=[0.30, 0.50, 0.20])
            k_ip = np.random.choice([3, 4, 5], p=[0.30, 0.50, 0.20])
        elif utype in ["account_takeover", "new_synthetic_user"]:
            k_fp = np.random.choice([2, 3], p=[0.60, 0.40])
            k_ip = np.random.choice([2, 3, 4], p=[0.40, 0.40, 0.20])
        elif utype == "cancellation_abuser":
            k_fp = np.random.choice([1, 2], p=[0.70, 0.30])
            k_ip = np.random.choice([1, 2], p=[0.70, 0.30])
        elif utype == "credit_bustout_user":
            k_fp = np.random.choice([1, 2], p=[0.60, 0.40])
            k_ip = np.random.choice([2, 3], p=[0.60, 0.40])
        else:
            k_fp, k_ip = 2, 2
        user_sticky_fps[u] = list(np.random.choice(all_fingerprints, size=k_fp, replace=False))
        user_sticky_ips[u] = list(np.random.choice(all_ips, size=k_ip, replace=False))

    IP_MASK_PROB = {
        "account_takeover":    0.35, "bot_booking": 0.30, "ring_operator": 0.25,
        "new_synthetic_user":  0.30, "credit_bustout_user": 0.10, "cancellation_abuser": 0.05, "legit": 0.00,
    }

    DATACENTER_ASN_PROB = {
        "account_takeover":    0.35, "bot_booking": 0.40, "ring_operator": 0.30,
        "new_synthetic_user":  0.30, "credit_bustout_user": 0.10, "cancellation_abuser": 0.05, "legit": 0.02,
    }

    DATACENTER_ASN_LIST = list(DATACENTER_ASN_RANGES)

    session_rows = []
    for i in range(n_bookings):
        u = booking_fact.loc[i, "user_id"]
        utype = user_master.loc[user_master["user_id"] == u, "user_fraud_type"].values[0]

        ip_masking = np.random.rand() < IP_MASK_PROB.get(utype, 0.0)

        if ip_masking:
            fp = np.random.choice(user_sticky_fps[u])
            ip = random_ip()
        elif u in fraud_user_set:
            if utype in ["ring_operator", "bot_booking"]:
                fp = np.random.choice(fraud_fp_pool) if np.random.rand() < 0.65 else np.random.choice(user_sticky_fps[u])
                ip = np.random.choice(fraud_ip_pool) if np.random.rand() < 0.65 else np.random.choice(user_sticky_ips[u])
            elif utype in ["account_takeover", "new_synthetic_user"]:
                fp = np.random.choice(fraud_fp_pool) if np.random.rand() < 0.55 else np.random.choice(user_sticky_fps[u])
                ip = np.random.choice(fraud_ip_pool) if np.random.rand() < 0.55 else np.random.choice(user_sticky_ips[u])
            else:
                fp = np.random.choice(user_sticky_fps[u])
                ip = np.random.choice(user_sticky_ips[u])
        else:
            fp = user_sticky_fps[u][0]
            ip = np.random.choice(user_sticky_ips[u])
            if np.random.rand() < 0.01:
                ip = np.random.choice(fraud_ip_pool)

        if ip_masking:
            asn = int(np.random.choice(DATACENTER_ASN_LIST))
        elif np.random.rand() < DATACENTER_ASN_PROB.get(utype, 0.02):
            asn = int(np.random.choice(DATACENTER_ASN_LIST))
        else:
            asn = int(np.random.choice(
                list(range(1000, 10000)) + list(range(15000, 30000)) + list(range(35000, 99999))
            ))

        vpn_proxy = 1 if asn in DATACENTER_ASN_RANGES else 0

        if u in fraud_user_set:
            if np.random.rand() < 0.70:
                ttb = int(np.clip(np.random.normal(45, 5), 15, 90))
            else:
                ttb = int(np.clip(np.random.lognormal(4.2, 0.6), 20, 600))
        else:
            ttb = int(np.clip(np.random.lognormal(5.8, 0.7), 60, 3600))

        session_rows.append({
            "session_id": make_uuid(),
            "booking_id": booking_fact.loc[i, "booking_id"],
            "device_fingerprint": fp,
            "ip_address": ip,
            "asn": asn,
            "is_vpn_or_proxy": vpn_proxy,
            "os": fp_to_os[fp],
            "browser": fp_to_browser[fp],
            "time_to_book_seconds": ttb,
        })

    session_context = pd.DataFrame(session_rows)

    # device_switch_flag
    _sc_user = session_context.merge(booking_fact[["booking_id", "user_id"]], on="booking_id")
    _user_fp_counts = _sc_user.groupby(["user_id", "device_fingerprint"]).size().reset_index(name="_cnt")
    _primary_fp = (_user_fp_counts.sort_values("_cnt", ascending=False)
                  .drop_duplicates("user_id")
                  .rename(columns={"device_fingerprint": "_primary_fp"})[["user_id", "_primary_fp"]])
    _sc_user = _sc_user.merge(_primary_fp, on="user_id")
    session_context["device_switch_flag"] = (_sc_user["device_fingerprint"] != _sc_user["_primary_fp"]).astype(int)

    # ============================================================
    # PASSENGER DETAILS
    # ============================================================
    THROWAWAY_SET = set(THROWAWAY_DOMAINS)
    fraud_indices_list = list(fraud_booking_set)
    nonfraud_indices_list = [i for i in range(n_bookings) if i not in fraud_booking_set]

    n_fraud_with_throwaway = int(0.70 * len(fraud_indices_list))
    n_nonfraud_with_throwaway = int(0.30 * len(nonfraud_indices_list))

    fraud_throwaway_set = set(np.random.choice(fraud_indices_list, size=n_fraud_with_throwaway, replace=False))
    nonfraud_throwaway_set = set(np.random.choice(nonfraud_indices_list, size=n_nonfraud_with_throwaway, replace=False))

    passenger_rows = []
    for i in range(n_bookings):
        bid = booking_fact.loc[i, "booking_id"]
        aid = booking_fact.loc[i, "agency_id"]
        n_pax = int(booking_fact.loc[i, "passengers_count"])
        is_fraud_booking = i in fraud_booking_set
        agency_domain = agency_to_domain[aid]
        will_have_throwaway = i in fraud_throwaway_set or i in nonfraud_throwaway_set

        pax_domains = []
        for p in range(n_pax):
            r = np.random.rand()
            if is_fraud_booking and will_have_throwaway:
                if p == 0: pax_domain = np.random.choice(THROWAWAY_DOMAINS)
                elif r < 0.15: pax_domain = agency_domain
                elif r < 0.50: pax_domain = np.random.choice(GENERIC_DOMAINS)
                else: pax_domain = np.random.choice(THROWAWAY_DOMAINS)
            elif is_fraud_booking:
                if r < 0.25: pax_domain = agency_domain
                else: pax_domain = np.random.choice(GENERIC_DOMAINS)
            elif will_have_throwaway:
                if p == 0: pax_domain = np.random.choice(THROWAWAY_DOMAINS)
                elif r < 0.60: pax_domain = agency_domain
                elif r < 0.88: pax_domain = np.random.choice(GENERIC_DOMAINS)
                else: pax_domain = np.random.choice(THROWAWAY_DOMAINS)
            else:
                if r < 0.75: pax_domain = agency_domain
                elif r < 0.97: pax_domain = np.random.choice(GENERIC_DOMAINS)
                else: pax_domain = np.random.choice(CORPORATE_DOMAIN_POOL)
            pax_domains.append(pax_domain)

        for d in pax_domains:
            passenger_rows.append({
                "passenger_id": make_uuid(),
                "booking_id": bid,
                "user_id": booking_fact.loc[i, "user_id"],
                "agency_id": aid,
                "passenger_email_domain": d,
                "is_suspicious_domain": int(d in THROWAWAY_SET),
            })
    passenger_details = pd.DataFrame(passenger_rows)

    # ============================================================
    # POST BOOKING EVENTS
    # ============================================================
    post_events = []
    for idx, row in booking_fact.iterrows():
        uid = row["user_id"]
        val = float(row["booking_value"])
        is_fraud_booking = idx in fraud_booking_set
        utype = user_master.loc[user_master["user_id"] == uid, "user_fraud_type"].values[0]

        cancel_p, dispute_p = 0.06, 0.004
        if is_fraud_booking: cancel_p, dispute_p = 0.12, 0.03
        if utype == "cancellation_abuser":
            cancel_p = 0.70 if not is_fraud_booking else 0.88
            dispute_p = 0.002
        
        if utype == "credit_bustout_user" and is_fraud_booking:
            cancel_p, dispute_p = 0.07, 0.18
            booking_fact.loc[idx, "route_type"] = "international"
            booking_fact.loc[idx, "lead_time_days"] = np.random.randint(0, 2)
            booking_fact.loc[idx, "booking_value"] = round(float(np.clip(np.random.normal(18000, 4000), 5000, 25000)), 2)
            booking_fact.loc[idx, "payment_method"] = "credit_line"
            val = float(booking_fact.loc[idx, "booking_value"])

        if utype == "new_synthetic_user" and is_fraud_booking:
            cancel_p, dispute_p = 0.14, 0.06
            booking_fact.loc[idx, "lead_time_days"] = np.random.randint(0, 4)

        is_cancelled = np.random.rand() < cancel_p
        is_disputed = (not is_cancelled) and (np.random.rand() < dispute_p)
        cancel_delay = int(np.random.randint(0, 3)) if is_cancelled else 0
        dispute_delay = int(np.random.randint(7, 45)) if is_disputed else 0

        chargeback_amount, final_loss_amount = 0.0, 0.0
        if is_disputed:
            if utype == "credit_bustout_user": cb_r, loss_r = np.random.uniform(0.85, 1.00), np.random.uniform(0.65, 0.95)
            elif utype in ["account_takeover", "new_synthetic_user"]: cb_r, loss_r = np.random.uniform(0.70, 0.95), np.random.uniform(0.45, 0.85)
            else: cb_r, loss_r = np.random.uniform(0.50, 0.90), np.random.uniform(0.25, 0.75)
            chargeback_amount = round(val * cb_r, 2)
            final_loss_amount = round(chargeback_amount * loss_r, 2)

        post_events.append({
            "event_id": f"E{str(idx+1).zfill(6)}", "booking_id": row["booking_id"],
            "is_cancelled": int(is_cancelled), "cancel_delay_days": cancel_delay,
            "is_disputed": int(is_disputed), "dispute_delay_days": dispute_delay,
            "chargeback_amount": chargeback_amount, "final_loss_amount": final_loss_amount
        })
    post_booking_events = pd.DataFrame(post_events)

    # ============================================================
    # BOOKING LABEL TABLE
    # ============================================================
    booking_label_table = pd.DataFrame({
        "booking_id": booking_fact["booking_id"], "fraud_label": 0, "fraud_reason": "legit"
    })
    booking_label_table.loc[list(fraud_booking_set), "fraud_label"] = 1

    for idx in fraud_booking_set:
        uid = booking_fact.loc[idx, "user_id"]
        utype = user_master.loc[user_master["user_id"] == uid, "user_fraud_type"].values[0]
        cancelled = post_booking_events.loc[idx, "is_cancelled"] == 1
        if utype == "cancellation_abuser" and cancelled: reason = "user has abnormally high cancellation rate"
        elif utype == "credit_bustout_user": reason = "credit bustout: high value international + high loss"
        elif utype == "new_synthetic_user": reason = "new user with risky infra + abnormal velocity"
        elif utype == "account_takeover": reason = "new device/ip + short lead time"
        elif utype == "ring_operator": reason = "shared infra across multiple fraud users"
        elif utype == "bot_booking": reason = "burst/automation pattern in activity"
        else: reason = "fraud user suspicious booking pattern"
        booking_label_table.loc[idx, "fraud_reason"] = reason

    # Save to out_dir if specified
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        agency_master.to_csv(os.path.join(out_dir, "agency_master.csv"), index=False)
        user_master.to_csv(os.path.join(out_dir, "user_master.csv"), index=False)
        booking_fact.to_csv(os.path.join(out_dir, "booking_fact.csv"), index=False)
        session_context.to_csv(os.path.join(out_dir, "session_context.csv"), index=False)
        passenger_details.to_csv(os.path.join(out_dir, "passenger_details.csv"), index=False)
        post_booking_events.to_csv(os.path.join(out_dir, "post_booking_events.csv"), index=False)
        booking_label_table.to_csv(os.path.join(out_dir, "booking_label_table.csv"), index=False)

    return {
        "agency_master": agency_master,
        "user_master": user_master,
        "booking_fact": booking_fact,
        "session_context": session_context,
        "passenger_details": passenger_details,
        "post_booking_events": post_booking_events,
        "booking_label_table": booking_label_table
    }

if __name__ == "__main__":
    # Default behavior: replicate the notebook's execution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_out_dir = os.path.join(base_dir, "Data", "Tables")
    
    print("Generating synthetic data...")
    dfs = data_generator(out_dir=target_out_dir)
    print(f"✅ Dataset generated successfully in: {target_out_dir}")
    for name, df in dfs.items():
        print(f" - {name}: {len(df)} rows")
