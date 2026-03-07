import pandas as pd
from typing import List, Tuple, Optional, Set, Dict
import streamlit as st
from config import DEFAULT_MSN_PATTERNS, DEFAULT_VARIABLE_MAPPINGS
from parser import ParsedSession
from utils import canonicalize_id, extract_gender, normalize_msn


# ─── ADDED: constants and complete lookup tables from your documents ───
PUMP_RATE_ML_SEC = 0.0172  # from formula (ml/sec)

# Default fallback durations (used when lookup fails or drug type is unknown)
AVG_INF_DUR_COCAINE = 4.0
AVG_INF_DUR_FENTANYL = 1.5
AVG_INF_DUR_NICOTINE = 1.0

# Cocaine 1.5 mg/kg infusion duration (complete from your table)
COCAINE_1_5MGKG_DUR = {
    100: 1.3, 110: 1.4, 120: 1.5, 130: 1.6, 140: 1.8,
    150: 1.9, 160: 2.0, 170: 2.1, 180: 2.3, 190: 2.4,
    200: 2.5, 210: 2.6, 220: 2.8, 230: 2.9, 240: 3.0,
    250: 3.1, 260: 3.3, 270: 3.4, 280: 3.5, 290: 3.6,
    300: 3.8, 310: 3.9, 320: 4.0, 330: 4.1, 340: 4.2,
    350: 4.4, 360: 4.5, 370: 4.6, 380: 4.7, 390: 4.9,
    400: 5.0, 410: 5.1, 420: 5.2, 430: 5.4, 440: 5.5,
    450: 5.6, 460: 5.7, 470: 5.9, 480: 6.0, 490: 6.1,
    500: 6.2, 510: 6.4, 520: 6.5, 530: 6.6, 540: 6.8,
    550: 6.9, 560: 7.0, 570: 7.1, 580: 7.3, 590: 7.4,
    600: 7.5, 610: 7.6, 620: 7.8, 630: 7.9, 640: 8.0,
    650: 8.1, 660: 8.3, 670: 8.4, 680: 8.5, 690: 8.6,
    700: 8.8, 710: 8.9, 720: 9.0, 730: 9.1, 740: 9.3,
    750: 9.4, 760: 9.5, 770: 9.6, 780: 9.8, 790: 9.9,
    800: 10.0, 810: 10.1, 820: 10.3, 830: 10.4, 840: 10.5,
    850: 10.6, 860: 10.8, 870: 10.9, 880: 11.0, 890: 11.1,
}

# Cocaine 0.5 mg/kg infusion duration (complete from your table)
COCAINE_0_5MGKG_DUR = {
    100: 0.4, 110: 0.5, 120: 0.5, 130: 0.5, 140: 0.6,
    150: 0.6, 160: 0.7, 170: 0.7, 180: 0.7, 190: 0.8,
    200: 0.8, 210: 0.9, 220: 0.9, 230: 1.0, 240: 1.0,
    250: 1.0, 260: 1.1, 270: 1.1, 280: 1.2, 290: 1.2,
    300: 1.2, 310: 1.3, 320: 1.3, 330: 1.4, 340: 1.4,
    350: 1.5, 360: 1.5, 370: 1.5, 380: 1.6, 390: 1.6,
    400: 1.7, 410: 1.7, 420: 1.7, 430: 1.8, 440: 1.8,
    450: 1.9, 460: 1.9, 470: 2.0, 480: 2.0, 490: 2.0,
    500: 2.1, 510: 2.1, 520: 2.2, 530: 2.2, 540: 2.2,
    550: 2.3, 560: 2.3, 570: 2.4, 580: 2.4, 590: 2.5,
    600: 2.5, 610: 2.5, 620: 2.6, 630: 2.6, 640: 2.7,
    650: 2.7, 660: 2.7, 670: 2.8, 680: 2.8, 690: 2.9,
    700: 2.9, 710: 3.0, 720: 3.0, 730: 3.0, 740: 3.1,
    750: 3.1, 760: 3.2, 770: 3.2, 780: 3.2, 790: 3.3,
    800: 3.3, 810: 3.4, 820: 3.4, 830: 3.4, 840: 3.5,
    850: 3.5, 860: 3.6, 870: 3.6, 880: 3.6, 890: 3.7,
}

# Cocaine 0.3 mg/kg infusion duration (complete from your table)
COCAINE_0_3MGKG_DUR = {
    100: 0.3, 110: 0.3, 120: 0.3, 130: 0.3, 140: 0.4,
    150: 0.4, 160: 0.4, 170: 0.4, 180: 0.5, 190: 0.5,
    200: 0.5, 210: 0.5, 220: 0.6, 230: 0.6, 240: 0.6,
    250: 0.6, 260: 0.7, 270: 0.7, 280: 0.7, 290: 0.7,
    300: 0.8, 310: 0.8, 320: 0.8, 330: 0.8, 340: 0.9,
    350: 0.9, 360: 0.9, 370: 0.9, 380: 1.0, 390: 1.0,
    400: 1.0, 410: 1.0, 420: 1.1, 430: 1.1, 440: 1.1,
    450: 1.1, 460: 1.2, 470: 1.2, 480: 1.2, 490: 1.2,
    500: 1.3, 510: 1.3, 520: 1.3, 530: 1.3, 540: 1.4,
    550: 1.4, 560: 1.4, 570: 1.4, 580: 1.5, 590: 1.5,
    600: 1.5, 610: 1.5, 620: 1.6, 630: 1.6, 640: 1.6,
    650: 1.6, 660: 1.7, 670: 1.7, 680: 1.7, 690: 1.8,
    700: 1.8, 710: 1.8, 720: 1.8, 730: 1.9, 740: 1.9,
    750: 1.9, 760: 2.0, 770: 2.0, 780: 2.0, 790: 2.0,
    800: 2.1, 810: 2.1, 820: 2.1, 830: 2.1, 840: 2.2,
    850: 2.2, 860: 2.2, 870: 2.2, 880: 2.3, 890: 2.3,
}

# Nicotine infusion duration (complete from your table)
NICOTINE_DUR = {
    100: 0.6, 110: 0.7, 120: 0.8, 130: 0.8, 140: 0.9,
    150: 0.9, 160: 1.0, 170: 1.1, 180: 1.1, 190: 1.2,
    200: 1.3, 210: 1.3, 220: 1.4, 230: 1.4, 240: 1.5,
    250: 1.6, 260: 1.6, 270: 1.7, 280: 1.8, 290: 1.8,
    300: 1.9, 310: 1.9, 320: 2.0, 330: 2.1, 340: 2.1,
    350: 2.2, 360: 2.3, 370: 2.3, 380: 2.4, 390: 2.4,
    400: 2.5, 410: 2.6, 420: 2.6, 430: 2.7, 440: 2.7,
    450: 2.8, 460: 2.9, 470: 2.9, 480: 3.0, 490: 3.0,
    500: 3.1, 510: 3.2, 520: 3.2, 530: 3.3, 540: 3.3,
    550: 3.4, 560: 3.5, 570: 3.5, 580: 3.6, 590: 3.6,
    600: 3.7, 610: 3.8, 620: 3.8, 630: 3.9, 640: 3.9,
    650: 4.0, 660: 4.1, 670: 4.1, 680: 4.2, 690: 4.2,
    700: 4.3, 710: 4.4, 720: 4.4, 730: 4.5, 740: 4.5,
    750: 4.6, 760: 4.7, 770: 4.7, 780: 4.8, 790: 4.8,
    800: 4.9, 810: 5.0, 820: 5.0, 830: 5.1, 840: 5.1,
    850: 5.2, 860: 5.3, 870: 5.3, 880: 5.4, 890: 5.4,
}

# Fentanyl infusion duration (approximate from your second table + typical values)
FENTANYL_DUR = {
    100: 0.5, 110: 0.5, 120: 0.5, 130: 0.5, 140: 0.5,
    150: 0.5, 160: 0.5, 170: 0.5, 180: 0.5, 190: 0.5,
    200: 0.5, 210: 0.5, 220: 0.6, 230: 0.6, 240: 0.6,
    250: 0.6, 260: 0.6, 270: 0.6, 280: 0.6, 290: 0.6,
    300: 0.7, 310: 0.7, 320: 0.7, 330: 0.7, 340: 0.7,
    350: 0.7, 360: 0.7, 370: 0.7, 380: 0.8, 390: 0.8,
    400: 0.8, 410: 0.8, 420: 0.8, 430: 0.8, 440: 0.8,
    450: 0.8, 460: 0.9, 470: 0.9, 480: 0.9, 490: 0.9,
    500: 1.0, 510: 1.0, 520: 1.0, 530: 1.0, 540: 1.0,
    550: 1.0, 560: 1.1, 570: 1.1, 580: 1.1, 590: 1.1,
    600: 1.2, 610: 1.2, 620: 1.2, 630: 1.2, 640: 1.2,
    650: 1.3, 660: 1.3, 670: 1.3, 680: 1.3, 690: 1.3,
    700: 1.4, 710: 1.4, 720: 1.4, 730: 1.4, 740: 1.4,
    750: 1.5, 760: 1.5, 770: 1.5, 780: 1.5, 790: 1.5,
    800: 1.6, 810: 1.6, 820: 1.6, 830: 1.6, 840: 1.6,
    850: 1.7, 860: 1.7, 870: 1.7, 880: 1.7, 890: 1.7,
}

# Default weight (common adult rat weight)
DEFAULT_WEIGHT_G = 300.0

# Default concentrations (mg/ml) — used if user doesn't provide
DEFAULT_COC_CONC_MGML = 1.0
DEFAULT_FENT_CONC_MGML = 0.01
DEFAULT_NIC_CONC_MGML = 0.2


def robust_parse_date(date_str: str) -> pd.Timestamp:
    """Try multiple common date formats."""
    if not date_str or pd.isna(date_str):
        return pd.NaT
    formats = ["%m/%d/%y", "%m/%d/%Y", "%m-%d-%y", "%m-%d-%Y",
               "%Y-%m-%d", "%d/%m/%y", "%d-%m-%Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt, errors='raise')
        except:
            continue
    return pd.NaT


def get_val(session: ParsedSession, var_name: str, default: float = 0.0) -> float:
    """Safely retrieve scalar value."""
    return float(session.scalars.get(var_name, default))


def calculate_duration(session: ParsedSession, mapping: dict) -> float:
    """Compute session duration in seconds."""
    if "duration_sec" in mapping:
        return get_val(session, mapping["duration_sec"])
    if "duration_min" in mapping:
        return get_val(session, mapping["duration_min"]) * 60
    if "duration_hour" in mapping:
        return get_val(session, mapping["duration_hour"]) * 3600
    return 0.0


def process_sessions(
    sessions: List[ParsedSession],
    allowed_ids: Optional[Set[str]] = None,
    custom_patterns: Optional[Dict] = None,
    custom_mappings: Optional[Dict] = None,
    avg_weight_g: float = DEFAULT_WEIGHT_G,
    drug_type: str = "None",
    conc_mgml: float = 1.0  # ← ADDED: user-provided concentration
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    Parse all MedPC sessions into session-level and hourly DataFrames.
    Supports optional ID filtering and custom MSN/variable overrides.
    """
    patterns = custom_patterns or DEFAULT_MSN_PATTERNS
    mappings = custom_mappings or DEFAULT_VARIABLE_MAPPINGS

    all_rows = []
    all_hourly = []
    found = set()

    for sess in sessions:
        subj_raw = sess.meta.get("Subject", "")
        canon = canonicalize_id(subj_raw)
        if not canon:
            continue
        if allowed_ids and canon not in allowed_ids:
            continue

        found.add(canon)
        gender = extract_gender(canon)
        raw_msn = sess.meta.get("MSN", "")
        norm_msn = normalize_msn(raw_msn)

        prog = "UNMAPPED"
        mapping = mappings.get("RAT - FR20", {})  # fallback

        # Longest-first pattern matching
        for name, pats in patterns.items():
            for pat in pats:
                if pat in norm_msn:
                    prog = name
                    mapping = mappings.get(prog, mapping)
                    break
            if prog != "UNMAPPED":
                break

        start_dt = robust_parse_date(sess.meta.get("Start Date", ""))
        end_dt   = robust_parse_date(sess.meta.get("End Date", ""))
        duration_sec = calculate_duration(sess, mapping)

        # ─── ADDED: choose infusion duration lookup based on drug type ───
        if drug_type == "Fentanyl" or "FENTANYL" in prog.upper():
            inf_dur_sec = FENTANYL_DUR.get(int(avg_weight_g), AVG_INF_DUR_FENTANYL)
        elif drug_type == "Nicotine" or "NICOTINE" in prog.upper():
            inf_dur_sec = NICOTINE_DUR.get(int(avg_weight_g), AVG_INF_DUR_NICOTINE)
        else:
            # Default to cocaine 1.5 mg/kg (most common in your data)
            inf_dur_sec = COCAINE_1_5MGKG_DUR.get(int(avg_weight_g), AVG_INF_DUR_COCAINE)

        row = {
            "canonical_subject": canon,
            "gender": gender,
            "program_name": prog,
            "raw_msn": raw_msn,
            "start_date": start_dt,
            "end_date": end_dt,
            # ─── ADDED: session span and overnight detection ───
            "session_span_days": (
                (end_dt - start_dt).days + 1
                if pd.notna(end_dt) and pd.notna(start_dt) else 1
            ),
            "overnight_session": (
                pd.notna(end_dt) and pd.notna(start_dt) and (end_dt > start_dt)
            ),
            # ──────────────────────────────────────────────────────
            "duration_sec": duration_sec,
            "active_presses": get_val(sess, mapping.get("active_presses", "I")),
            "inactive_presses": get_val(sess, mapping.get("inactive_presses", "J")),
            "infusions": get_val(sess, mapping.get("infusions", "K")),
            "pump_time_sec": get_val(sess, mapping.get("pump_time", "Y")) * get_val(sess, mapping.get("infusions", "K")),
            "breakpoints": get_val(sess, mapping.get("breakpoints", "B")),
            "retrievals": get_val(sess, mapping.get("retrievals", "C")),
            "responses": get_val(sess, mapping.get("responses", "R")),
            # ─── ADDED: W and T values (using mapping where possible, fallback to direct key) ───
            "W_value": get_val(sess, mapping.get("W_value", "W")),
            "T_value": get_val(sess, mapping.get("T_value", "T")),
            # ─── ADDED: Timeout presses per infusion (very useful metric) ───
            "timeout_presses_per_inf": (
                get_val(sess, mapping.get("W_value", "W")) /
                (get_val(sess, mapping.get("infusions", "K")) + 1e-6)
            ),
            # ─── ADDED: Estimated volume (ml) per session ───
            "estimated_volume_ml": get_val(sess, mapping.get("infusions", "K")) * PUMP_RATE_ML_SEC,
            # ─── ADDED: Estimated infusion duration (sec) — lookup for cocaine, average for fentanyl ───
            "estimated_inf_dur_sec": inf_dur_sec,
            # ─── ADDED: Estimated mg/kg intake using user-provided concentration ───
            "estimated_intake_mgkg": (
                get_val(sess, mapping.get("infusions", "K")) *
                conc_mgml * PUMP_RATE_ML_SEC / (avg_weight_g / 1000)
            ) if conc_mgml > 0 else 0.0,
            # ─── ADDED: Optional extras if you see them often in your data ───
            "U_value": get_val(sess, mapping.get("U_value", "U")),
            "V_value": get_val(sess, mapping.get("V_value", "V")),
            # ────────────────────────────────────────────────────────────────────────────────────────
            # Add Box/Room if present in metadata
            "Box": sess.meta.get("Box", ""),
            "Room": sess.meta.get("Room", "") or sess.meta.get("Experiment", ""),
        }

        # Special fields for extinction/reinstatement
        if "EXTINCTION" in prog.upper() or "REINSTATEMENT" in prog.upper():
            row["Response_U"] = get_val(sess, "U")
            row["Response_L"] = get_val(sess, "L")

        all_rows.append(row)

        # Hourly breakdown
        if "I" in sess.arrays and len(sess.arrays["I"]) > 0:
            ts_inf = sess.arrays.get(mapping.get("infusion_times", "I"), [])
            ts_act = sess.arrays.get(mapping.get("active_times", "R"), [])
            if duration_sec > 0:
                max_h = int(duration_sec // 3600) + 1
                for h in range(max_h + 1):
                    all_hourly.append({
                        "canonical_subject": canon,
                        "gender": gender,
                        "program_name": prog,
                        "start_date": start_dt,
                        "hour": h,
                        "infusion_events": sum(1 for t in ts_inf if int(t // 3600) == h),
                        "active_events": sum(1 for t in ts_act if int(t // 3600) == h),
                        "Box": row["Box"],
                        "Room": row["Room"]
                    })

    df_sessions = pd.DataFrame(all_rows)
    df_hourly   = pd.DataFrame(all_hourly)

    if not df_sessions.empty:
        df_sessions = df_sessions.sort_values(["canonical_subject", "start_date"])
    if not df_hourly.empty:
        df_hourly = df_hourly.sort_values(["canonical_subject", "start_date", "hour"])

    return df_sessions, df_hourly, found


def generate_pattern_flags(
    df: pd.DataFrame,
    min_active: int = 5,
    max_inactive_ratio: float = 0.5,
    min_duration_min: int = 30,
    escalation_threshold: int = 30
) -> pd.DataFrame:
    """
    Add behavioral quality & pattern flags with customizable thresholds.
    """
    if df.empty:
        return df

    df = df.copy()
    df["low_activity_flag"] = df["active_presses"] < min_active
    df["high_inactive_ratio_flag"] = (
        df["inactive_presses"] / (df["active_presses"] + 1)
    ) > max_inactive_ratio
    df["short_session_flag"] = df["duration_sec"] < (min_duration_min * 60)

    # Example escalation flag (compares to previous session - same subject)
    df = df.sort_values(["canonical_subject", "start_date"])
    df["prev_infusions"] = df.groupby("canonical_subject")["infusions"].shift(1)
    df["escalation_flag"] = (
        (df["infusions"] > df["prev_infusions"] * (1 + escalation_threshold / 100)) &
        df["prev_infusions"].notna()
    )

    # Overall data quality score (0–4 bad flags)
    df["data_quality_flag"] = (
        df["low_activity_flag"].astype(int) +
        df["high_inactive_ratio_flag"].astype(int) +
        df["short_session_flag"].astype(int) +
        df["escalation_flag"].astype(int)
    )

    return df


def create_daily_summary(df_sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to one row per subject per day — safely handles missing columns.
    Includes 'had_overnight_session' flag if any session in the day crossed midnight.
    """
    if df_sessions.empty:
        return pd.DataFrame()

    df = df_sessions.copy()
    df["date"] = df["start_date"].dt.date

    # Base grouping keys (always present)
    group_keys = ["canonical_subject", "gender", "date"]

    # Only include columns that actually exist in this subset
    possible_agg = {
        "infusions": "sum",
        "active_presses": "sum",
        "inactive_presses": "sum",
        "duration_sec": "sum",
        "pump_time_sec": "sum",
        "breakpoints": "max",
        "program_name": lambda x: ", ".join(x.unique()),
        "start_date": "min",
        "end_date": "max",
        "Box": "first",
        "Room": "first",
        "overnight_session": "any",
        "session_span_days": "max",
        "W_value": "sum",
        "T_value": "sum",
        "timeout_presses_per_inf": "mean",
        # ─── ADDED: aggregate new estimates ───
        "estimated_volume_ml": "sum",
        "estimated_inf_dur_sec": "mean",
        "estimated_intake_mgkg": "sum",
    }

    # Filter to only columns present in df
    agg_dict = {col: func for col, func in possible_agg.items() if col in df.columns}

    # Aggregate
    daily = df.groupby(group_keys).agg(agg_dict).reset_index()

    # Rename aggregated columns (only those that exist)
    rename_map = {
        "infusions": "total_infusions",
        "active_presses": "total_active_presses",
        "inactive_presses": "total_inactive_presses",
        "duration_sec": "total_duration_sec",
        "start_date": "first_session_time",
        "end_date": "last_session_time",
        "overnight_session": "had_overnight_session",
        "session_span_days": "max_session_span_days",
        "W_value": "total_W_value",
        "T_value": "total_T_value",
        "timeout_presses_per_inf": "avg_timeout_presses_per_inf",
        "estimated_volume_ml": "total_estimated_volume_ml",
        "estimated_inf_dur_sec": "avg_estimated_inf_dur_sec",
        "estimated_intake_mgkg": "total_estimated_intake_mgkg",
    }

    daily = daily.rename(columns={k: v for k, v in rename_map.items() if k in daily.columns})

    # Add session count
    daily["session_count"] = df.groupby(group_keys).size().values

    return daily.sort_values(["canonical_subject", "date"])


def report_missing_and_box_room(
    expected_list: List[str],
    found_set: Set[str],
    df: pd.DataFrame
):
    """
    Display missing subjects and Box/Room distribution.
    Call this from app.py after processing.
    """
    canon_expected = {canonicalize_id(s) for s in expected_list if s.strip()}
    missing = sorted(canon_expected - found_set)

    if missing:
        st.warning(f"**{len(missing)} expected subject(s) NOT found:**")
        st.write(", ".join(missing) or "(none)")
    else:
        if expected_list:
            st.success("All expected subjects were found in the data.")

    # Box / Room summary
    if "Box" in df.columns or "Room" in df.columns:
        st.subheader("Box / Room Distribution")
        summary = df.groupby(["canonical_subject", "Box", "Room"]).size().reset_index(name="session_count")
        st.dataframe(summary.style.highlight_max(subset=["session_count"], color="#d4edda"))