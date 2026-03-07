from typing import Dict, List, Any

# ============================================================================
# LYNCH LAB MEDPC ANALYZER - CONFIGURATION FILE (v5.9 - W & T ADDED)
# ============================================================================
# INTERMITTENT ACCESS remains FIRST and very specific — this fixes missing days.
# Now includes W (timeout presses, side responses) and T (trials/counts) in mappings.
# ============================================================================

METADATA_KEYS = [
    "start date", "end date", "subject", "msn", "experiment", "group",
    "box", "start time", "end time", "time unit", "room", "cage"
]

# ────────────────────────────────────────────────
# Variable letter mappings per program type
# ────────────────────────────────────────────────
map_rat_fr = {
    "infusions": "I",
    "active_presses": "R",
    "inactive_presses": "L",
    "infusion_timestamps": "J",
    "active_timestamps": "K",
    "inactive_timestamps": [],
    "duration": "Z",
    "extra_vars": ["W"],
    "W_value": "W",          # timeout / non-reinforced presses
    "T_value": "T",          # trial count or total attempts
}

map_rat_fent = {
    "infusions": "I",
    "active_presses": "R",
    "inactive_presses": "L",
    "infusion_timestamps": "J",
    "active_timestamps": "K",
    "inactive_timestamps": [],
    "duration": "Z",
    "extra_vars": ["W"],
    "special_processing": "J_ARRAY_HOURLY",
    "W_value": "W",          # timeout presses during infusion delay
    "T_value": "T",          # often total reinforcer attempts
}

map_rat_int = {
    "infusions": "I",
    "active_presses": "R",
    "inactive_presses": "L",
    "infusion_timestamps": "L",
    "active_timestamps": "P",
    "inactive_timestamps": [],
    "duration": "Z",
    "extra_vars": ["U"],
    "W_value": "W",          # presses during inter-trial interval
    "T_value": "T",          # number of lever extensions / trials
}

map_rat_pr = {
    "infusions": "I",
    "active_presses": "R",
    "inactive_presses": "L",
    "infusion_timestamps": "J",
    "active_timestamps": "K",
    "inactive_timestamps": [],
    "duration": "Z",
    "breakpoint": "V",
    "W_value": "W",          # timeout presses or side responses
    "T_value": "T",          # total lever presses or attempts
}

map_rat_ext = {
    "active_presses": "R",
    "inactive_presses": "L",
    "active_timestamps": "J",
    "inactive_timestamps": "K",
    "duration": "Z",
    "special_processing": "EXTINCTION_DETAIL",
    "W_value": "W",          # responses during extinction timeout
    "T_value": "T",          # extinction trials or cue presentations
}

map_rat_cue = {
    "infusions": "N",
    "active_presses": "R",
    "inactive_presses": "L",
    "active_timestamps": "J",
    "inactive_timestamps": [],
    "duration": "Z",
    "W_value": "W",          # cue-induced responses
    "T_value": "T",          # cue presentations or trials
}

map_rat_food = {
    "active_presses": "R",
    "inactive_presses": "L",
    "reinforcers": "I",
    "duration": "Z",
    "W_value": "W",          # food magazine entries or side presses
    "T_value": "T",
}

map_flush = {
    "pump_time": "I",
    "duration": "Z",
    "W_value": "W",          # any responses during flush
    "T_value": "T",
}

map_withdrawal = {
    "duration": "Z",
    "W_value": "W",          # withdrawal-related responses
    "T_value": "T",
}

map_mouse = {
    "infusions": "R",
    "active_presses": "A",
    "inactive_presses": "I",
    "infusion_timestamps": "G",
    "active_timestamps": [],
    "inactive_timestamps": [],
    "duration": "Z",
    "extra_vars": ["L", "G"],
    "W_value": "W",
    "T_value": "T",
}

map_mouse_pr = {
    "infusions": "R",
    "active_presses": "A",
    "inactive_presses": "I",
    "infusion_timestamps": [],
    "active_timestamps": [],
    "inactive_timestamps": [],
    "duration": "Z",
    "breakpoint": "V",
    "extra_vars": ["L", "G"],
    "W_value": "W",
    "T_value": "T",
}

# ────────────────────────────────────────────────
# MSN pattern matching (program name → list of normalized substrings)
# ────────────────────────────────────────────────
DEFAULT_MSN_PATTERNS: Dict[str, List[str]] = {
    # === INTERMITTENT ACCESS — FIRST AND VERY SPECIFIC ===
    "RAT - INTERMITTENT ACCESS": [
        "newintermittentaccessldesd",
        "intermittentaccessldesd",
        "2025newintermittentaccess",
        "3newintermittentaccess",
        "4newintermittentaccess",
        "g136anewintermittentaccess",
        "g136bnewintermittentaccess",
        "shortinta",
        "newintermittentaccessldfoodrestrictesd",
        "intermittentaccess",
        "intaccess",
        "intermittentld",
        "accessldesd",
        "intermittent access ld",
        "new intermittent access",
        "g136a new intermittent",
        "g136b new intermittent",
        # Keep "intermittent" last — it's broad, so lower priority
        "intermittent",
        "ld intermittent",
        "intermittentldesd",
        "new intermittent",
    ],

    # === FENTANYL FR40 LD FOOD RESTRICT — make sure it's very specific ===
    "RAT - FENTANYL FR40 LD FOOD RESTRICT": [
        "fentanyl1secfr40ldfoodrestrictesd",
        "g136afentanyl1secfr40ldfoodrestrictesd",
        "g136bfentanyl1secfr40ldfoodrestrictesd",
        # Add any common variants you see
        "fentanyl 1 sec fr40 ld food restrict esd",
        "fentanyl1secfr40ld foodrestrictesd",
    ],

    # === FENTANYL FR40 LD (non-food restrict) ===
    "RAT - FENTANYL FR40 LD": [
        "fentanyl1secfr40ldesd",
        "g136afentanyl1secfr40ldesd",
        "g136bfentanyl1secfr40ldesd",
        "fentanyl 1 sec fr40 ld esd",
        "fentanyl1secfr40esd",
        "fentanyl fr40 esd",
    ],

    # === FR FOOD / MAG TRAINING ===
    "RAT - FR FOOD / MAG TRAINING": [
        "2025newfrfoodtrain",
        "frfoodtrain",
        "newfrfoodtrain",
        "g136anewfrfoodtrain",
        "g136bnewfrfoodtrain",
        "g13614bnewfrfoodtrain",
        "frfood",
        "dt4",
        "new frfood train",
    ],

    # === WITHDRAWAL — move higher if it's being stolen by intermittent ===
    "RAT - WITHDRAWAL": [
        "withdrawalldesd",
        "g136awithdrawalldesd",
        "g136bwithdrawalldesd",
        "withdrawal ld esd",
        "withdrawal",
        "g136awithdrawal",
        "g136bwithdrawal",
        "withdrawalld",
    ],
    "RAT - CONTINUOUS FENTANYL": ["continuousfentanyl"],
    "RAT - FR20": ["fr20", "g136afr20", "fr20esd", "fr20pdt"],
    "RAT - FR40": ["fr40", "g136afr40"],
    "RAT - PR COCAINE": ["prcocaine", "prcocaineesd", "g136aprcocaine"],
    "RAT - PR FENTANYL": ["prfent", "g136aprfent", "prfentesd"],
    "RAT - EXTINCTION": [
        "extinct",
        "extinction",
        "extinctmustextby9",
        "g136aextinct",
        "g140aextinct",
        "extinctionreinstatement",
        "2008 v6 to 10 ext plus cue",
        "b boxes extinct must ext by 9",
        "extinct-reinstate",
        "z test extinct",
        "g136aprocaine",
        "g136aboxes",
    ],
    "RAT - REINSTATEMENT": [
        "reinstate",
        "onlyrein",
        "g136aonlyrein",
        "reinstatementg140aboxes2017",
        "extinctionreinstatementg140",
        "g136areinstate",
    ],
    "RAT - CUE RELAPSE 7HR": [
        "g138acuerelapse7hrpreathold",
        "g138a",
        "g136acuerelapse",
        "g140acuerelapse7hr",
        "cuerelapse",
        "g138acuerelapsenohold2025",
        "g140a cue relapse 7hr pretx hold",
        "copy of g140a cue relapse",
        "relapse esd",
    ],
    "RAT - CUE RELAPSE 2HR": [
        "g140acuerelapsefollowing2hr",
        "test cue relapse following 2hr",
        "testcuerelapsefollowing2hr",
        "relapseesd",
    ],
    "RAT - DISCRETE TRIAL": ["dt4final", "g136adt4final"],
    "RAT - FLUSH": ["flush", "g136aflush", "withdrawalmpc"],
    "RAT - WITHDRAWAL": ["withdrawal", "g136awithdrawal", "g136bwithdrawal", "withdrawalld", "withdrawalldesd"],
    "MOUSE - EXTENDED ACCESS": [
        "mouseextendedaccess",
        "mouseextendedaccessv2",
        "mouseintera",
        "mouseintermittentaccess",
    ],
    "MOUSE - PR": ["mousepr", "mouse pr"],
    "MOUSE - FR1": ["mousefr1", "mouse fr1"],
}

# ────────────────────────────────────────────────
# Program name → variable mapping
# ────────────────────────────────────────────────
DEFAULT_VARIABLE_MAPPINGS: Dict[str, Dict[str, Any]] = {
    "RAT - INTERMITTENT ACCESS": map_rat_int,
    "RAT - FR FOOD / MAG TRAINING": map_rat_food,
    "RAT - FENTANYL FR40 LD": map_rat_fent,
    "RAT - FENTANYL FR40 LD FOOD RESTRICT": map_rat_fent,
    "RAT - FENTANYL FR40": map_rat_fent,
    "RAT - CONTINUOUS FENTANYL": map_rat_fent,
    "RAT - FR20": map_rat_fr,
    "RAT - FR40": map_rat_fr,
    "RAT - PR COCAINE": map_rat_pr,
    "RAT - PR FENTANYL": map_rat_pr,
    "RAT - EXTINCTION": map_rat_ext,
    "RAT - REINSTATEMENT": map_rat_ext,
    "RAT - CUE RELAPSE 7HR": map_rat_cue,
    "RAT - CUE RELAPSE 2HR": map_rat_cue,
    "RAT - DISCRETE TRIAL": map_rat_fr,
    "RAT - FLUSH": map_flush,
    "RAT - WITHDRAWAL": map_withdrawal,
    "MOUSE - EXTENDED ACCESS": map_mouse,
    "MOUSE - PR": map_mouse_pr,
    "MOUSE - FR1": map_mouse,
}

def normalize_msn(msn: str) -> str:
    if not msn:
        return ""
    return (
        str(msn)
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
    )


print("✅ config.py v5.9 loaded — W & T values now mapped, Intermittent Access still prioritized!")