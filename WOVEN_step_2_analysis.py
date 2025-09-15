import pandas as pd

# === File paths ===
csv_file = "Data/WOVEN_Pt_Case_Report_cleaned.csv"

# === Load cleaned data ===
df = pd.read_csv(csv_file)

# === 1. Case Type distribution ===
print("\n--- Case Type Distribution ---")
print(df["Case Type"].value_counts(dropna=False))

# === 2. Other useful quick stats ===

# Length of Pt. Case Description (tokens, chars)
df["desc_len_tokens"] = df["Pt. Case Description"].str.split().apply(len)
df["desc_len_chars"] = df["Pt. Case Description"].str.len()

print("\n--- Pt. Case Description Length (tokens) ---")
print(df["desc_len_tokens"].describe())

print("\n--- Pt. Case Description Length (characters) ---")
print(df["desc_len_chars"].describe())

# Case Created timeline
df["Case Created"] = pd.to_datetime(df["Case Created"], errors="coerce")
print("\n--- Case Created Timeline ---")
print(df["Case Created"].dt.to_period("M").value_counts().sort_index())

# Optional: export summary to CSV
summary = {
    "case_type_distribution": df["Case Type"].value_counts(dropna=False).to_dict(),
    "desc_len_tokens": df["desc_len_tokens"].describe().to_dict(),
    "desc_len_chars": df["desc_len_chars"].describe().to_dict(),
    "cases_per_month": df["Case Created"].dt.to_period("M").value_counts().sort_index().to_dict()
}
pd.Series(summary).to_json("Data/WOVEN_Pt_Case_Report_stats.json", indent=2)

print("\nSummary JSON saved to WOVEN_Pt_Case_Report_stats.json")
