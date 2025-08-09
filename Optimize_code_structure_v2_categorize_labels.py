import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from sk import OPENAI_API_KEY  # your local API key file

# ====== CONFIG ======
INPUT_FILE = "EPPC_output_json/input_1_cleaned_node_names.json"
OUTPUT_FILE = "EPPC_output_json/classified_intents.json"
MODEL_NAME = "gpt-4o-mini"  # fast + reasoning
# ====================

# Load input data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize LLM
llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)

# Prompt template — curly braces escaped
prompt_template = ChatPromptTemplate.from_template("""
You are given a label from a hierarchical clinical communication annotation scheme.
The hierarchy has 3 levels: "code" (broadest), "subcode" (mid-level), and "subsubcode" (most specific).
Each label is given with its full hierarchy and index.

There are two intent types:
1. **Interactional Intent (Communicative Function)** – the discourse role or communicative function
   (e.g., question, inform, request, greeting), similar to dialogue acts in DailyDialog or ISO 24617-2.
2. **Goal-Oriented Intent (Semantic Purpose)** – the domain-specific task, outcome, or information goal
   (e.g., refill prescription, schedule appointment), similar to CLINC150 or SNIPS.

**Task:** Classify the given label into one of these two types and briefly explain your reasoning.

Respond strictly in JSON:
{{
  "type": "Interactional" | "Goal-Oriented",
  "explanation": "short reason"
}}

Hierarchy info:
Code level: "{code_label}"
Subcode level: "{subcode_label}"
Subsubcode level: "{subsubcode_label}"
Current label to classify: "{current_label}"
""")

def classify_label(code_label, subcode_label, subsubcode_label, current_label):
    prompt = prompt_template.format_messages(
        code_label=code_label,
        subcode_label=subcode_label,
        subsubcode_label=subsubcode_label,
        current_label=current_label
    )
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        return json.loads(content)
    except Exception as e:
        return {
            "type": "ERROR",
            "explanation": f"Parsing error: {e}, Raw: {response.content if 'response' in locals() else ''}"
        }

# Helper to get parent label
def find_parent_label(level_list, idx_prefix):
    for item in level_list:
        idx, lbl = item.split(":", 1)
        if idx.strip() == idx_prefix.strip():
            return lbl.strip()
    return ""

classified_data = {"code": [], "subcode": [], "subsubcode": []}

# Classify codes
for item in data.get("code", []):
    idx, label = item.split(":", 1)
    result = classify_label(label, "", "", label)
    classified_data["code"].append({
        "index": idx.strip(),
        "label": label.strip(),
        "type": result.get("type"),
        "explanation": result.get("explanation")
    })

# Classify subcodes
for item in data.get("subcode", []):
    idx, label = item.split(":", 1)
    code_idx = "_".join(idx.strip().split("_")[:1])
    code_label = find_parent_label(data.get("code", []), code_idx)
    result = classify_label(code_label, label, "", label)
    classified_data["subcode"].append({
        "index": idx.strip(),
        "label": label.strip(),
        "type": result.get("type"),
        "explanation": result.get("explanation")
    })

# Classify subsubcodes
for item in data.get("subsubcode", []):
    idx, label = item.split(":", 1)
    code_idx = "_".join(idx.strip().split("_")[:1])
    subcode_idx = "_".join(idx.strip().split("_")[:2])
    code_label = find_parent_label(data.get("code", []), code_idx)
    subcode_label = find_parent_label(data.get("subcode", []), subcode_idx)
    result = classify_label(code_label, subcode_label, label, label)
    classified_data["subsubcode"].append({
        "index": idx.strip(),
        "label": label.strip(),
        "type": result.get("type"),
        "explanation": result.get("explanation")
    })

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(classified_data, f, indent=2, ensure_ascii=False)

print(f"Classification completed. Output saved to {OUTPUT_FILE}")
