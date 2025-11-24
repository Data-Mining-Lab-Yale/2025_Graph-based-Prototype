import json
import pandas as pd

TARGET_PROJECT_NAME = "EPPC" #TARGET_PROJECT_NAME +"_output_json/
# TARGET_PROJECT_NAME = "PV"

# === File paths ===
codebook_path = TARGET_PROJECT_NAME +"_output_json/codebook_hierarchy.json"
frequency_summary_path = TARGET_PROJECT_NAME +"_output_json/annotation_code_frequency_summary_corrected.csv"
output_path = TARGET_PROJECT_NAME +"_output_json/codebook_coverage_with_frequencies.csv"

# === Load codebook ===
with open(codebook_path, "r", encoding="utf-8") as f:
    codebook = json.load(f)

# === Load frequency summary ===
df_freq = pd.read_csv(frequency_summary_path)

# === Build codebook DataFrame, skipping empty or missing IDs ===
codebook_entries = []
for node in codebook.get("nodes", []):
    code_id = node.get("id", "").strip()
    if code_id:  # Only keep non-empty IDs
        codebook_entries.append({
            "Matched Codebook Label": code_id,
            "Level": node.get("type", "Unknown")
        })
df_codebook = pd.DataFrame(codebook_entries)

# === Merge codebook with annotation frequency data ===
df_merged = df_codebook.merge(
    df_freq, 
    on=["Matched Codebook Label", "Level"], 
    how="left"
)

# === Fill missing frequencies with 0 and clean result ===
df_merged["Frequency"] = df_merged["Frequency"].fillna(0).astype(int)

# === Remove rows with empty or null labels after merge (if any) ===
df_merged = df_merged[df_merged["Matched Codebook Label"].notna()]
df_merged = df_merged[df_merged["Matched Codebook Label"].str.strip() != ""]

# === Sort and save ===
df_merged = df_merged.sort_values(by="Frequency", ascending=False)
df_merged.to_csv(output_path, index=False)

print(f"✅ Saved cleaned coverage file: {output_path}")
