import json
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from tqdm import tqdm
from sk import my_sk  # your key file

INPUT = "EPPC_output_json/node_names_by_type_with_index.json"

# ---------- FIX: normalize the input file ----------
def load_index_label_map(path: str) -> dict:
    """
    Reads node_names_by_type_with_index.json which has:
      { "code": ["1: ...", ...], "subcode": ["1_1: ...", ...], "subsubcode": ["1_1_1: ...", ...] }
    Returns a flat mapping: { "1": "...", "1_1": "...", "1_1_1": "..." }.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    idx2label = {}
    for level in ("code", "subcode", "subsubcode"):
        for item in raw.get(level, []):
            if ":" not in item:
                # skip malformed rows
                continue
            idx, label = item.split(":", 1)
            idx2label[idx.strip()] = label.strip()
    return idx2label

idx2label = load_index_label_map(INPUT)
print(f"Loaded {len(idx2label)} labels from {INPUT}")

# ---------- OPTIONAL: deduplicate subsubcodes (same parent + same label keeps lowest suffix) ----------
def dedupe_subsubcodes(idx2label: dict):
    keep = {}
    alias = {}
    by_parent_and_label = defaultdict(list)

    for idx, label in idx2label.items():
        parts = idx.split("_")
        if len(parts) == 3:
            parent = "_".join(parts[:2])
            by_parent_and_label[(parent, label)].append(idx)
        else:
            keep[idx] = label  # pass through code and subcode

    for (parent, label), indices in by_parent_and_label.items():
        # choose canonical = lowest numeric suffix
        def suffix(i): return int(i.split("_")[-1])
        canonical = sorted(indices, key=suffix)[0]
        keep[canonical] = label
        for i in indices:
            alias[i] = canonical

    return keep, alias

idx2label_dedup, alias_map = dedupe_subsubcodes(idx2label)
print(f"After dedupe: {len(idx2label)} -> {len(idx2label_dedup)} (subsubcode duplicates collapsed)")

# ---------- LLM setup ----------
llm = ChatOpenAI(api_key=my_sk, model="gpt-4o-mini", temperature=0, max_tokens=256)

# classify each label as Interactional vs Goal-Oriented
classification = {}
rationale_log = {}
for idx, label in tqdm(idx2label_dedup.items(), desc="Classifying"):
    prompt = f'''You are classifying label *texts* from a healthcare dialogue taxonomy.

Label: "{label}"

Choose one:
- Interactional Intent (communicative function; dialog acts like request, inform, confirm, greet)
- Goal-Oriented Intent (semantic purpose; domain goals like refill, scheduling, diagnostics)

Return exactly:
Label Type: [Interactional or Goal-Oriented]
Explanation: <brief rationale>
'''
    resp = llm([HumanMessage(content=prompt)]).content
    try:
        ltype = resp.split("Label Type:")[1].split("\n")[0].strip()
        expl  = resp.split("Explanation:")[1].strip()
    except Exception:
        ltype, expl = "Unclear", resp.strip()

    classification[idx] = {"label": label, "type": ltype}
    rationale_log[idx] = {"label": label, "type": ltype, "explanation": expl}

# ---------- rebuild hierarchy ----------
# Top level = Interactional; Sublevel = Goal-Oriented grouped under nearest interactional by simple heuristic.
# You can swap in a better grouper later.
interactional = [i for i, v in classification.items() if v["type"].lower().startswith("interactional")]
goal_oriented = [i for i, v in classification.items() if v["type"].lower().startswith("goal")]

# Assign new top-level indices 1..N in a stable order
interactional_sorted = sorted(interactional, key=lambda i: classification[i]["label"].lower())
new_top_index = {}
optimized = []
for n, idx in enumerate(interactional_sorted, start=1):
    new_top_index[idx] = f"{n}"
    optimized.append(f'{n}: {classification[idx]["label"]}')

# Attach goal-oriented under a parent using a cheap lexical match; else under "Other"
def pick_parent(goal_label: str):
    for idx in interactional_sorted:
        if classification[idx]["label"].lower() in goal_label.lower():
            return idx
    return None

# Ensure Other bucket exists if needed
other_parent_idx = None

children_counts = defaultdict(int)
index_mapping = {}  # old label-index/string → new index
# map top-level originals to new indices
for idx in interactional_sorted:
    index_mapping[idx] = new_top_index[idx]

for idx in goal_oriented:
    goal_label = classification[idx]["label"]
    p = pick_parent(goal_label)
    if p is None:
        # create Other if missing
        if other_parent_idx is None:
            # allocate next top index
            next_top = str(len(new_top_index) + 1)
            other_parent_idx = f"other_{next_top}"
            new_top_index[other_parent_idx] = next_top
            optimized.append(f"{next_top}: Other")
        parent_new = new_top_index[other_parent_idx]
    else:
        parent_new = new_top_index[p]

    children_counts[parent_new] += 1
    new_idx = f"{parent_new}_{children_counts[parent_new]}"
    optimized.append(f"{new_idx}: {goal_label}")
    index_mapping[idx] = new_idx

# also carry through any code/subcode that LLM labeled as Unclear just to not lose them
unclear = [i for i, v in classification.items() if v["type"] == "Unclear"]
if unclear:
    next_top = str(len(new_top_index) + 1)
    optimized.append(f"{next_top}: Unclear")
    for k, idx in enumerate(sorted(unclear), start=1):
        new_idx = f"{next_top}_{k}"
        optimized.append(f"{new_idx}: {classification[idx]['label']}")
        index_mapping[idx] = new_idx

# ---------- persist ----------
with open("EPPC_output_json/optimized_label_structure.json", "w", encoding="utf-8") as f:
    json.dump(optimized, f, indent=2, ensure_ascii=False)

with open("EPPC_output_json/index_mapping.json", "w", encoding="utf-8") as f:
    json.dump(index_mapping, f, indent=2, ensure_ascii=False)

with open("EPPC_output_json/llm_classification_log.json", "w", encoding="utf-8") as f:
    json.dump(rationale_log, f, indent=2, ensure_ascii=False)

with open("EPPC_output_json/alias_map.json", "w", encoding="utf-8") as f:
    json.dump(alias_map, f, indent=2, ensure_ascii=False)

print("Done:")
print(" - optimized_label_structure.json")
print(" - index_mapping.json")
print(" - llm_classification_log.json")
print(" - alias_map.json  (old subsubcode → canonical)")
