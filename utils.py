import re
import pandas as pd

def canonicalize_id(subject_id: str) -> str:
    """Normalizes Subject IDs to prevent 0/O confusion and whitespace issues."""
    if pd.isna(subject_id) or str(subject_id).strip() == "":
        return ""
    return re.sub(r"^O", "0", str(subject_id).strip().upper())

def extract_gender(subject_id: str) -> str:
    """Infers gender from the last letter of the subject ID."""
    if not subject_id:
        return "Unknown"
    last = str(subject_id).strip().lower()[-1]
    return "Female" if last == "f" else "Male" if last == "m" else "Unknown"

def normalize_msn(msn: str) -> str:
    """Strips punctuation and spaces from MSN strings for reliable matching."""
    if pd.isna(msn):
        return ""
    return re.sub(r"[^\w]", "", str(msn)).lower()