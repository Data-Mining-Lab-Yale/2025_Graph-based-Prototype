import pandas as pd
import json
import re
import os

# === CONFIG ===
PROJECT_FILE_EPPC = "EPPC_codebook_04.07.2025.xlsx"
PROJECT_FILE_PV = "PV_Codebook.xlsx"
# excel_path = "Data/"+PROJECT_FILE_PV  # Excel file location
excel_path = "Data/"+PROJECT_FILE_EPPC  # Excel file location
sheet_name = "CodeBookMap"
# sheet_name = "EPPC_codebook"
output_dir = "EPPC_output_json"                    # Folder to save output
output_file = "codebook_hierarchy.json"       # Output filename

# === CREATE OUTPUT FOLDER IF NEEDED ===
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# === PREP ===
nodes = {}
links = []

def normalize(text):
    return re.sub(r'\s+', '_', text.strip()) if isinstance(text, str) else None

# === PARSE CODE BLOCKS ===
current_code = None
current_subcode = None

for idx, row in df.iterrows():
    code = row['CODE']
    code_def = row['CODE DEFINITION']
    subcode = row['SUB-CODE']
    subcode_def = row['SUB-CODE DEFINITION']
    
    if pd.notna(code):
        current_code = normalize(code)
        if current_code not in nodes:
            nodes[current_code] = {
                "id": current_code,
                "type": "code",
                "description": str(code_def).strip() if pd.notna(code_def) else ""
            }

    if pd.notna(subcode):
        if ":" in subcode:
            name, desc = map(str.strip, subcode.split(":", 1))
        else:
            name, desc = subcode.strip(), ""

        subcode_id = normalize(name)
        current_subcode = subcode_id
        
        if subcode_id not in nodes:
            nodes[subcode_id] = {
                "id": subcode_id,
                "type": "subcode",
                "description": desc
            }

        links.append({"source": current_code, "target": subcode_id})

    if pd.notna(subcode_def) and ":" in subcode_def:
        matches = re.findall(r'([^:\n]+):\s*([^:\n]+)', subcode_def)
        for name, desc in matches:
            subsub_id = normalize(name)
            if subsub_id not in nodes:
                nodes[subsub_id] = {
                    "id": subsub_id,
                    "type": "subsubcode",
                    "description": [desc.strip()]
                }
            else:
                if desc.strip() not in nodes[subsub_id]["description"]:
                    nodes[subsub_id]["description"].append(desc.strip())

            links.append({"source": current_subcode, "target": subsub_id})

# === FORMAT FINAL STRUCTURE ===
output = {
    "nodes": list(nodes.values()),
    "links": links
}

# === SAVE TO FILE IN ASSIGNED FOLDER ===
output_path = os.path.join(output_dir, output_file)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Saved to {output_path}")
