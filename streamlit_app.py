import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
import zipfile
import json
from datetime import datetime

# ============================================================================
# 1. CONFIGURATION & DEFAULTS
# ============================================================================

st.set_page_config(page_title="Lynch Lab MedPC", page_icon="🧬", layout="wide")

METADATA_KEYS = ["start date", "end date", "subject", "msn", "experiment", 
                 "group", "box", "start time", "end time", "time unit"]

# DEFAULT PATTERNS (Identical to Desktop App v4.6)
DEFAULT_MSN_PATTERNS = {
    "RAT - FR20": ["fr20"], "RAT - FR40": ["fr40"], "RAT - FR FOOD": ["frfood", "2025newfrfoodtrain", "newfrfoodtrain"],
    "RAT - FENTANYL FR40": ["fentanyl1secfr40ldesd"], "RAT - FENTANYL FR40 (FOOD RESTRICT)": ["fentanyl1secfr40ldfoodrestrictesd"],
    "RAT - INTERMITTENT ACCESS": ["newintermittentaccessldesd"], "RAT - WITHDRAWAL": ["withdrawalldesd"], 
    "RAT - INT ACCESS (FOOD RESTRICT)": ["newintermittentaccessldfoodrestrictesd"],
    "RAT - EXTINCTION FR20": ["g136afr20"], "RAT - EXTINCTION PROCAINE": ["g136aprocaine"], 
    "RAT - EXTINCTION BOXES": ["g136aboxes"], "RAT - REINSTATEMENT": ["g136areinstate"],
    "RAT - CUE RELAPSE A": ["g138acuerelapse7hrpreathold", "g138a"], "RAT - CUE RELAPSE B": ["g138bcuerelapse7hrpretxhold", "g138b", "cuerelapse"],
    "RAT - PR COCAINE": ["prcocaine"], "RAT - PR FENTANYL": ["prfent", "prfentesd"],
    "MOUSE - EXTENDED ACCESS": ["mouseextendedaccess", "mouseextendedaccessv2", "mouseintera", "mouseintermittentaccess"], 
    "MOUSE - PR": ["mousepr", "mouse pr"], "MOUSE - FR1": ["mousefr1", "mouse fr1"]
}

# Variable Definitions
map_rat_fr = {"infusions": ["I"], "active_presses": ["R"], "inactive_presses": [], "infusion_timestamps": ["J"], "active_timestamps": ["K"], "inactive_timestamps": [], "breakpoint": "K", "duration": "Z", "extra_vars": ["W"]}
map_rat_int = {"infusions": ["I"], "active_presses": ["U"], "inactive_presses": ["R"], "infusion_timestamps": ["F", "G"], "active_timestamps": ["L", "P"], "inactive_timestamps": ["M", "D"], "breakpoint": "Q", "duration": "Z", "extra_vars": ["W"]}
map_rat_fent = {"infusions": ["I"], "active_presses": ["R"], "inactive_presses": ["A"], "infusion_timestamps": ["J"], "active_timestamps": ["J"], "inactive_timestamps": ["J"], "breakpoint": None, "duration": "Z", "special_processing": "J_ARRAY_HOURLY", "extra_vars": ["W"]}
map_rat_cue = {"infusions": ["N"], "active_presses": ["R"], "inactive_presses": ["M"], "infusion_timestamps": ["N"], "active_timestamps": ["A", "D", "F", "G"], "inactive_timestamps": ["H", "I", "J", "K"], "breakpoint": "Q", "duration": "Z", "extra_vars": ["W"]}
map_rat_pr = {"infusions": ["I"], "active_presses": ["R"], "inactive_presses": ["A"], "infusion_timestamps": ["J"], "active_timestamps": ["J"], "inactive_timestamps": ["J"], "breakpoint": "V", "duration": "Z", "extra_vars": ["W"]}
map_rat_ext = {"infusions": ["N"], "active_presses": ["U", "M"], "inactive_presses": ["P"], "infusion_timestamps": [], "active_timestamps": [], "inactive_timestamps": [], "breakpoint": None, "duration": "Z", "special_extraction": "EXTINCTION_DETAIL"}
map_mouse = {"infusions": ["R"], "active_presses": ["A"], "inactive_presses": ["I"], "infusion_timestamps": ["G"], "active_timestamps": [], "inactive_timestamps": [], "breakpoint": None, "duration": "Z", "extra_vars": ["L", "G"]}
map_mouse_pr = {"infusions": ["R"], "active_presses": ["A"], "inactive_presses": ["I"], "infusion_timestamps": [], "active_timestamps": [], "inactive_timestamps": [], "breakpoint": "V", "duration": "Z", "extra_vars": ["L", "G"]}

DEFAULT_VARIABLE_MAPPINGS = {
    "RAT - FR20": map_rat_fr, "RAT - FR40": map_rat_fr, "RAT - FR FOOD": map_rat_fr,
    "RAT - FENTANYL FR40": map_rat_fent, "RAT - FENTANYL FR40 (FOOD RESTRICT)": map_rat_fent,
    "RAT - INTERMITTENT ACCESS": map_rat_int, "RAT - WITHDRAWAL": map_rat_int, "RAT - INT ACCESS (FOOD RESTRICT)": map_rat_int,
    "RAT - EXTINCTION FR20": map_rat_ext, "RAT - EXTINCTION PROCAINE": map_rat_ext, "RAT - EXTINCTION BOXES": map_rat_ext, "RAT - REINSTATEMENT": map_rat_ext,
    "RAT - CUE RELAPSE A": map_rat_cue, "RAT - CUE RELAPSE B": map_rat_cue,
    "RAT - PR COCAINE": map_rat_pr, "RAT - PR FENTANYL": map_rat_pr,
    "MOUSE - EXTENDED ACCESS": map_mouse, "MOUSE - PR": map_mouse_pr, "MOUSE - FR1": map_mouse
}

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def canonicalize_id(subject_id):
    if pd.isna(subject_id) or str(subject_id).strip() == "": return ""
    return re.sub(r"^O", "0", str(subject_id).strip().upper())

def extract_gender(subject_id):
    if not subject_id: return "Unknown"
    last = str(subject_id).strip().lower()[-1]
    return "Female" if last == "f" else "Male" if last == "m" else "Unknown"

def normalize_msn(msn):
    if pd.isna(msn): return ""
    return re.sub(r"[^\w]", "", str(msn)).lower()

def extract_array_data(lines, start_idx):
    data = []
    i = start_idx + 1
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r"^[a-zA-Z]:", line) or (":" in line and line.split(":")[0].lower() in METADATA_KEYS): break
        if re.match(r"^\d+:", line):
            try:
                values = [float(x) for x in line.split(":", 1)[1].split()]
                data.extend(values)
            except ValueError: pass
        i += 1
    return data

def calculate_duration(arrays, scalars, key="Z", time_unit="seconds"):
    if key in scalars: return scalars[key]
    vals = arrays.get(key, [])
    if len(vals) >= 3: return vals[0]*3600 + vals[1]*60 + vals[2]
    if len(vals) >= 1: return vals[0] * 60 if str(time_unit).lower() == "minutes" else vals[0]
    return 0

def get_data_quality_flags(row):
    flags = []
    if row['infusions'] == 0 and row['duration_sec'] > 3600: flags.append("No infusions despite long session")
    if row['active_presses'] == 0 and row['infusions'] > 0: flags.append("Infusions without active presses")
    if row['duration_sec'] == 0: flags.append("Missing or zero duration")
    return "; ".join(flags) if flags else "OK"

# ============================================================================
# 3. PLOTTING FUNCTION
# ============================================================================

def create_plot(data, x, y, title, hue=None, kind="bar", palette="Set1", errorbar=None, style=None, markers=True):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    if kind == "bar":
        sns.barplot(data=data, x=x, y=y, hue=hue, errorbar=errorbar, palette=palette)
    elif kind == "line":
        sns.lineplot(data=data, x=x, y=y, hue=hue, style=style, markers=markers, palette=palette)
        plt.xticks(rotation=45)
    elif kind == "scatter":
        sns.scatterplot(data=data, x=x, y=y, hue=hue, size=style, alpha=0.7, palette=palette)
    elif kind == "box":
        sns.boxplot(data=data, x=x, y=y, hue=hue, palette=palette)
        sns.stripplot(data=data, x=x, y=y, color='black', alpha=0.3)
        
    plt.title(title)
    plt.tight_layout()
    
    # Save to BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return buf

# ============================================================================
# 4. MAIN APP UI
# ============================================================================

st.title("🧬 Lynch Lab MedPC Analyzer")
st.markdown("Upload your raw MedPC files to generate Excel reports and visualizations instantly.")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Upload Config (Optional)")
    settings_file = st.file_uploader("Settings.json (Overrides defaults)", type=["json"])
    
    st.header("2. Upload ID List")
    id_file = st.file_uploader("Allowed IDs (.txt)", type=["txt"])
    
    st.header("3. Upload Data")
    data_files = st.file_uploader("MedPC Raw Files", accept_multiple_files=True)

# --- PROCESSING ---
if st.button("Run Analysis", type="primary"):
    if not id_file or not data_files:
        st.error("Please upload both an ID List and at least one Data File.")
    else:
        # 1. LOAD SETTINGS
        msn_patterns = DEFAULT_MSN_PATTERNS.copy()
        variable_mappings = DEFAULT_VARIABLE_MAPPINGS.copy()
        
        if settings_file:
            try:
                s = json.load(settings_file)
                if "msn_patterns" in s: msn_patterns = s["msn_patterns"]
                if "variable_mappings" in s: variable_mappings = s["variable_mappings"]
                st.success("✅ Custom Settings loaded.")
            except: st.warning("⚠️ Error loading Settings.json. Using defaults.")

        # 2. LOAD IDs
        allowed_ids = {line.decode("utf-8").strip() for line in id_file}
        allowed_ids_canon = {canonicalize_id(x) for x in allowed_ids}
        st.info(f"Loaded {len(allowed_ids)} Allowed IDs.")

        # 3. PROCESS FILES
        all_sess, all_hr = [], []
        found_ids = set()
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(data_files):
            # Update Progress
            progress = (idx + 1) / len(data_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {uploaded_file.name}")

            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            raw_sessions = re.split(r"(Start Date:)", content)
            sessions = [raw_sessions[i] + raw_sessions[i+1] for i in range(1, len(raw_sessions), 2) if i+1 < len(raw_sessions)]

            for sess_block in sessions:
                lines = sess_block.splitlines()
                if len(lines) < 5: continue
                
                # Parse Meta & Scalars
                meta, scalars = {}, {}
                for line in lines:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k_clean = k.strip().lower()
                        if k_clean in METADATA_KEYS: meta[k_clean] = v.strip()
                        if k_clean == "msn": meta["msn"] = v.strip()
                        if len(k_clean) == 1 and k_clean.isalpha():
                            try: scalars[k_clean.upper()] = float(v.strip())
                            except: pass

                if "subject" not in meta: continue
                canon = canonicalize_id(meta["subject"])
                if canon not in allowed_ids_canon: continue
                found_ids.add(canon)

                # Determine Program
                msn_norm = normalize_msn(meta.get("msn", ""))
                prog = "Unknown"
                for p_name, patterns in msn_patterns.items():
                    if any(pat in msn_norm for pat in patterns):
                        prog = p_name
                        break
                if prog == "Unknown": continue

                mapping = variable_mappings.get(prog, {})
                arrays = {}
                for i, line in enumerate(lines):
                    if re.match(r"^[A-Z]:", line.strip()):
                        arrays[line.strip()[0]] = extract_array_data(lines, i)

                # Calculations (Scalar Priority)
                def get_val(keys):
                    total = 0
                    for k in keys:
                        if k in scalars and scalars[k] > 0: total += scalars[k]
                        elif k in arrays: total += sum(arrays[k])
                    return total

                inf = get_val(mapping.get("infusions", []))
                act = get_val(mapping.get("active_presses", []))
                inact = get_val(mapping.get("inactive_presses", []))
                dur = calculate_duration(arrays, scalars, mapping.get("duration", "Z"), meta.get("time unit", "seconds"))
                
                bp = 0
                bp_key = mapping.get("breakpoint")
                if bp_key and bp_key in arrays and inf > 0:
                    bp_vals = arrays[bp_key]
                    if len(bp_vals) >= int(inf): bp = bp_vals[int(inf)-1]

                row = {
                    "source_file": uploaded_file.name, "subject": meta["subject"], "canonical_subject": canon, 
                    "gender": extract_gender(meta["subject"]),
                    "start_date": pd.to_datetime(meta["start date"], errors='coerce'), "program_name": prog,
                    "infusions": inf, "active_presses": act, "inactive_presses": inact, 
                    "duration_sec": dur, "efficiency_ratio": inf/act if act > 0 else 0
                }
                row["data_quality_flag"] = get_data_quality_flags(row)
                
                for extra in mapping.get("extra_vars", []): row[f"Value_{extra}"] = get_val([extra])
                if mapping.get("special_extraction") == "EXTINCTION_DETAIL":
                    for l in list("UMABCDEFGHIJKL"): row[f"Response_{l}"] = get_val([l])
                
                all_sess.append(row)

                # Hourly Logic
                if mapping.get("special_processing") == "J_ARRAY_HOURLY" and "J" in arrays:
                    j = arrays["J"]
                    for i in range(0, len(j), 7):
                        if i+6 < len(j):
                            all_hr.append({"canonical_subject": canon, "gender": row["gender"], "program_name": prog, 
                                           "hour": int(j[i]), "infusion_events": j[i+2], "active_events": j[i+1], "inactive_events": j[i+4]})
                else:
                    ts_inf = [t for k in mapping.get("infusion_timestamps", []) for t in arrays.get(k, []) if t>=0]
                    ts_act = [t for k in mapping.get("active_timestamps", []) for t in arrays.get(k, []) if t>=0]
                    ts_inact = [t for k in mapping.get("inactive_timestamps", []) for t in arrays.get(k, []) if t>=0]
                    
                    if meta.get("time unit", "").lower() == "minutes":
                        ts_inf, ts_act, ts_inact = [[t*60 for t in l] for l in [ts_inf, ts_act, ts_inact]]

                    max_h = int(dur // 3600)
                    all_ts = ts_inf + ts_act + ts_inact
                    if all_ts: max_h = max(max_h, int(max(all_ts)//3600))
                    
                    for h in range(max_h + 1):
                        all_hr.append({
                            "canonical_subject": canon, "gender": row["gender"], "program_name": prog, "hour": h,
                            "infusion_events": sum(1 for t in ts_inf if int(t//3600)==h),
                            "active_events": sum(1 for t in ts_act if int(t//3600)==h),
                            "inactive_events": sum(1 for t in ts_inact if int(t//3600)==h)
                        })

        # 4. RESULTS
        if not all_sess:
            st.warning("No matching data found.")
        else:
            df_sess = pd.DataFrame(all_sess)
            df_hr = pd.DataFrame(all_hr) if all_hr else pd.DataFrame()
            
            # Prepare ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                
                for prog in df_sess["program_name"].unique():
                    safe_prog = re.sub(r"[^A-Za-z0-9]", "_", prog)
                    sess_sub = df_sess[df_sess["program_name"] == prog].copy()
                    hr_sub = df_hr[df_hr["program_name"] == prog].copy() if not df_hr.empty else pd.DataFrame()
                    
                    # Aggregations
                    daily = sess_sub.groupby(["canonical_subject", "gender", "start_date"]).agg({
                        "infusions": "sum", "active_presses": "sum", "inactive_presses": "sum", "duration_sec": "sum"
                    }).reset_index()
                    daily.rename(columns={"infusions": "total_infusions", "active_presses": "total_active_presses", "inactive_presses": "total_inactive_presses"}, inplace=True)
                    
                    avgs = sess_sub.groupby(["canonical_subject", "gender"]).agg({
                        "infusions": ["mean", "std"], "active_presses": ["mean"], "duration_sec": "mean"
                    }).reset_index()
                    avgs.columns = ['_'.join(col).strip() if col[1] else col[0] for col in avgs.columns.values]

                    sessions_per = sess_sub.groupby(["canonical_subject", "gender"]).size().reset_index(name="n_sessions")
                    quality = sess_sub[sess_sub["data_quality_flag"] != "OK"][["canonical_subject", "start_date", "data_quality_flag"]]

                    # 1. SAVE EXCEL TO ZIP
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        sess_sub.to_excel(writer, "01_Session_Data", index=False)
                        if not hr_sub.empty: hr_sub.to_excel(writer, "02_Hourly_Data", index=False)
                        daily.to_excel(writer, "03_Daily_Summaries", index=False)
                        avgs.to_excel(writer, "04_Subject_Averages", index=False)
                        sessions_per.to_excel(writer, "06_Sessions_Per_Subject", index=False)
                        quality.to_excel(writer, "07_Data_Quality_Flags", index=False)
                    zf.writestr(f"{safe_prog}_Analysis.xlsx", excel_buffer.getvalue())

                    # 2. GENERATE & SAVE PLOTS TO ZIP
                    plot_list = [
                        (create_plot(hr_sub, "hour", "infusion_events", "Hourly Infusions", "canonical_subject"), "01_Hourly_Infusions"),
                        (create_plot(hr_sub, "hour", "active_events", "Hourly Active", "canonical_subject"), "02_Hourly_Active"),
                        (create_plot(daily, "start_date", "total_infusions", "Daily Infusions", "canonical_subject", kind="line"), "04_Daily_Infusions"),
                        (create_plot(daily, "start_date", "total_active_presses", "Daily Active", "canonical_subject", kind="line"), "05_Daily_Active"),
                        (create_plot(avgs, "canonical_subject", "infusions_mean", "Avg Infusions", "canonical_subject"), "07_Avg_Infusions"),
                        (create_plot(sess_sub, "active_presses", "infusions", "Efficiency", "gender", kind="scatter", style="duration_sec"), "06b_Efficiency"),
                        (create_plot(daily, "start_date", "total_active_presses", "Trajectory", "canonical_subject", kind="line", style="gender"), "12_Trajectory")
                    ]
                    
                    for plt_buf, name in plot_list:
                        zf.writestr(f"Plots/{safe_prog}/{name}.png", plt_buf.getvalue())

            # FINALIZE
            st.balloons()
            st.success("Analysis Complete!")
            
            st.download_button(
                label="📥 Download All Results (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"MedPC_Results_{datetime.now().strftime('%Y%m%d')}.zip",
                mime="application/zip"
            )
            
            # Show summary stats on screen
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Files Processed", len(data_files))
            with col2:
                st.metric("Unique IDs Found", len(found_ids))