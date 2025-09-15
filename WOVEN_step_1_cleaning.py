import pandas as pd

# === File paths ===
input_file = "Data/WOVEN_Pt_Case_Report_2023-2025.xlsx"
sheet_name = "printcsvreports - 20250825_13-2"
csv_out = "Data/WOVEN_Pt_Case_Report_cleaned.csv"
json_out = "Data/WOVEN_Pt_Case_Report_cleaned.json"

# === Load data ===
df = pd.read_excel(input_file, sheet_name=sheet_name)

# === Step 1: remove rows where "Pt. Case Description" is empty ===
cleaned = df.dropna(subset=["Pt. Case Description"]).copy()

# === Step 2: add new columns with default values ===
cleaned["patient-to-provider"] = 0
cleaned["provider-to-patient"] = 0
cleaned["provider-to-provider"] = 1
cleaned["is_telephone_note"] = 0

# === Step 3: save outputs ===
cleaned.to_csv(csv_out, index=False)
cleaned.to_json(json_out, orient="records", lines=True)

print(f"Cleaned CSV saved to {csv_out}")
print(f"Cleaned JSON saved to {json_out}")
