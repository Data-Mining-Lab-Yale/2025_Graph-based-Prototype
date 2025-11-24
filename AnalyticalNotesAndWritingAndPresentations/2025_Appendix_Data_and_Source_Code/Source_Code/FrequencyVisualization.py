import pandas as pd
import matplotlib.pyplot as plt


TARGET_PROJECT_NAME = "EPPC" #TARGET_PROJECT_NAME +"_output_json/
# TARGET_PROJECT_NAME = "PV"

# === Load your corrected frequency file ===
df = pd.read_csv(TARGET_PROJECT_NAME +"_output_json/annotation_code_frequency_summary_corrected.csv")

# === Standardize level formatting ===
df["Level"] = df["Level"].str.strip().str.capitalize()

# === Adjustable: How many top items to visualize ===
TOP_N = 20

# === Color assignment for each level ===
level_to_color = {
    "Code": "steelblue",
    "Subcode": "orange",
    "Subsubcode": "green",
    "Unknown": "gray"
}

# === 1. Mixed Top-N with color by level ===
top_n_df = df.sort_values(by="Frequency", ascending=False).head(TOP_N)
colors = top_n_df["Level"].map(lambda lvl: level_to_color.get(lvl, "gray"))

plt.figure(figsize=(14, 7))
plt.bar(top_n_df["Matched Codebook Label"], top_n_df["Frequency"], color=colors)
plt.title(f"Top {TOP_N} Annotation Codes (Mixed Levels)")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.legend(handles=[
    plt.Rectangle((0,0),1,1,color="steelblue", label="Code"),
    plt.Rectangle((0,0),1,1,color="orange", label="Subcode"),
    plt.Rectangle((0,0),1,1,color="green", label="Subsubcode"),
    plt.Rectangle((0,0),1,1,color="gray", label="Unknown")
])
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(TARGET_PROJECT_NAME +"_output_json/"+f"top{TOP_N}_mixed_by_corrected_levels.png")

# === 2. Top-N Codes Only ===
top_codes = df[df["Level"] == "Code"].sort_values(by="Frequency", ascending=False).head(TOP_N)
plt.figure(figsize=(14, 7))
plt.bar(top_codes["Matched Codebook Label"], top_codes["Frequency"], color="steelblue")
plt.title(f"Top {TOP_N} Codes Only")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(TARGET_PROJECT_NAME +"_output_json/"+f"top{TOP_N}_codes_only_corrected.png")

# === 3. Top-N Subcodes Only ===
top_subcodes = df[df["Level"] == "Subcode"].sort_values(by="Frequency", ascending=False).head(TOP_N)
plt.figure(figsize=(14, 7))
plt.bar(top_subcodes["Matched Codebook Label"], top_subcodes["Frequency"], color="orange")
plt.title(f"Top {TOP_N} Subcodes Only")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(TARGET_PROJECT_NAME +"_output_json/"+f"top{TOP_N}_subcodes_only_corrected.png")

# === 4. Top-N Sub-subcodes Only ===
top_subsubcodes = df[df["Level"] == "Subsubcode"].sort_values(by="Frequency", ascending=False).head(TOP_N)
plt.figure(figsize=(14, 7))
plt.bar(top_subsubcodes["Matched Codebook Label"], top_subsubcodes["Frequency"], color="green")
plt.title(f"Top {TOP_N} Sub-subcodes Only")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(TARGET_PROJECT_NAME +"_output_json/"+f"top{TOP_N}_subsubcodes_only_corrected.png")

print("✅ Saved all visualizations.")
