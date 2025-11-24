import json
from pathlib import Path

# Inputs and outputs
BETHESDA_PATH = Path("Data/Bethesda_data/Bethesda_message_dataset.jsonl")
OUT_PATH = Path("Bethesda_output/Bethesda_processed_messages_with_annotations.json")

# Set to True if you want to keep start/end span positions in each annotation
KEEP_SPANS = False

def convert_bethesda_to_processed(bethesda_jsonl_path, out_json_path, keep_spans=False):
    out = []
    with open(bethesda_jsonl_path, "r", encoding="utf-8") as f:
        for msg_idx, line in enumerate(f):
            rec = json.loads(line)
            context = rec.get("context", "")
            anns_in = rec.get("annotations", []) or []

            anns_out = []
            for k, a in enumerate(anns_in):
                ann_obj = {
                    "text_id": f"{msg_idx}_{k}",
                    "text": a.get("text", ""),
                    "code": a.get("codes", []) or []
                }
                if keep_spans:
                    # Preserve original span info as optional fields
                    if "start" in a:
                        ann_obj["start"] = a["start"]
                    if "end" in a:
                        ann_obj["end"] = a["end"]
                anns_out.append(ann_obj)

            out.append({
                "message": context,
                "message_id": msg_idx,
                "annotations": anns_out
            })

    with open(out_json_path, "w", encoding="utf-8") as w:
        json.dump(out, w, ensure_ascii=False, indent=2)

    print(f"Converted {len(out)} messages → {out_json_path}")

if __name__ == "__main__":
    convert_bethesda_to_processed(BETHESDA_PATH, OUT_PATH, keep_spans=KEEP_SPANS)
