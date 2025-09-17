import os
import json
import pandas as pd

# === File paths ===
csv_file = "Data/WOVEN/WOVEN_Pt_Case_Report_cleaned.csv"
out_dir = "Data/WOVEN"
os.makedirs(out_dir, exist_ok=True)
summary_json = os.path.join(out_dir, "WOVEN_Pt_Case_Report_stats.json")
mismatch_csv = os.path.join(out_dir, "WOVEN_is_multiple_mismatch.csv")

# === Load cleaned data ===
df = pd.read_csv(csv_file)

# === 1) Case Type distribution ===
print("\n--- Case Type Distribution ---")
print(df["Case Type"].value_counts(dropna=False))

# === 2) Text length stats ===
desc = df["Pt. Case Description"].fillna("")
df["desc_len_tokens"] = desc.str.split().apply(len)
df["desc_len_chars"] = desc.str.len()

print("\n--- Pt. Case Description Length (tokens) ---")
print(df["desc_len_tokens"].describe())

print("\n--- Pt. Case Description Length (characters) ---")
print(df["desc_len_chars"].describe())

# === 3) Timeline ===
df["Case Created"] = pd.to_datetime(df["Case Created"], errors="coerce")
print("\n--- Case Created Timeline ---")
cases_per_month_series = df["Case Created"].dt.to_period("M").value_counts().sort_index()
print(cases_per_month_series)

# Serialize months as strings for JSON
cases_per_month = {str(k): int(v) for k, v in cases_per_month_series.items()}

# === 4) Flag coverage and consistency checks ===
flag_cols = [
    "patient-to-provider",
    "provider-to-patient",
    "provider-to-provider",
    "is_telephone_note",
    "is_content",
    "is_multiple",
]
defaults = {
    "patient-to-provider": 0,
    "provider-to-patient": 0,
    "provider-to-provider": 1,
    "is_telephone_note": 1,
    "is_content": 1,
    "is_multiple": 1,
}
for c in flag_cols:
    if c not in df.columns:
        df[c] = defaults[c]
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(defaults[c]).astype(int)

role_cols = ["patient-to-provider", "provider-to-patient", "provider-to-provider"]
df["role_sum"] = df[role_cols].sum(axis=1)
df["multi_from_roles"] = (df["role_sum"] > 1).astype(int)

flag_counts = {c: int(df[c].sum()) for c in flag_cols}
print("\n--- Flag counts ---")
for k, v in flag_counts.items():
    print(f"{k}: {v}")

df["is_multiple_matches_roles"] = (df["is_multiple"] == df["multi_from_roles"]).astype(int)
mismatches = df[df["is_multiple"] != df["multi_from_roles"]].copy()
print(f"\nMismatches between is_multiple and role based multi label: {len(mismatches)}")
mismatches.to_csv(mismatch_csv, index=False)
if len(mismatches):
    print(f"Saved mismatch rows to {mismatch_csv}")

# === 5) Save compact summary JSON ===
summary = {
    "case_type_distribution": {str(k): int(v) for k, v in df["Case Type"].value_counts(dropna=False).items()},
    "desc_len_tokens": {str(k): (float(v) if hasattr(v, "item") else v) for k, v in df["desc_len_tokens"].describe().items()},
    "desc_len_chars": {str(k): (float(v) if hasattr(v, "item") else v) for k, v in df["desc_len_chars"].describe().items()},
    "cases_per_month": cases_per_month,  # month keys are strings now
    "flag_counts": flag_counts,
    "multi_label_rate_from_roles": float(df["multi_from_roles"].mean()) if len(df) else 0.0,
    "is_multiple_agreement_rate": float(df["is_multiple_matches_roles"].mean()) if len(df) else 0.0,
    "mismatch_rows": int(len(mismatches)),
}

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\nSummary JSON saved to {summary_json}")
