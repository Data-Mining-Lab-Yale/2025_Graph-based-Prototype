# Reduced_Sample_1.py
import argparse, json, random, re, os
from collections import defaultdict, Counter
import pandas as pd
from pathlib import Path
random.seed(13)

# -----------------------
# Quick-start command you can copy/paste
# (Edit paths if needed.)
# -----------------------
DEFAULT_CMD = (
    "python Reduced_Sample_1.py "
    "--config labels_config.json "
    "--interactional_json EPPC_output_json/subsentence_interactional_label.json "
    "--goal_json EPPC_output_json/subsentence_goal_oriented_label_filtered.json "
    "--topics_csv Data/topics_top20_words.csv "
    "--outdir toysets_out "
    "--per_label 60 --dev_frac 0.15 --test_frac 0.15"
)

# -----------------------
# Helpers
# -----------------------
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_text(s):
    return (s or "").strip()

def extract_selected_examples(raw_dict, selected_labels):
    """
    Fallback extraction (no aliases): aggregate all selected labels assigned to a span.
    Returns: list of {id, text, labels}
    """
    rows_by_id = {}
    for sid, entry in raw_dict.items():
        text = normalize_text(entry.get("text", ""))
        labs_entries = entry.get("labels", [])
        labs = []
        for le in labs_entries:
            lbls = le.get("label", [])
            if isinstance(lbls, str):
                lbls = [lbls]
            for lab in lbls:
                if lab in selected_labels:
                    labs.append(lab)
        if labs:
            labs = sorted(list(set(labs)))
            rows_by_id[sid] = {"id": sid, "text": text, "labels": labs}
    return list(rows_by_id.values())

# ---------- alias-aware extraction + verifier ----------
def compile_aliases(label_aliases):
    """
    Build matchers for each target label.
    Returns: list of (target_label, match_fn)
    """
    compiled = []
    for target, spec in label_aliases.items():
        exact = set([str(s).lower() for s in spec.get("raw_exact", [])])
        patt = [re.compile(p, flags=re.I) for p in spec.get("raw_patterns", [])]
        def make_fn(exact_set, patt_list):
            def fn(raw):
                r = (raw or "").lower()
                if r in exact_set:
                    return True
                for pr in patt_list:
                    if pr.search(raw or ""):
                        return True
                return False
            return fn
        compiled.append((target, make_fn(exact, patt)))
    return compiled

def extract_selected_examples_with_aliases(raw_dict, target_labels, label_aliases,
                                           verify_bucket=None):
    """
    Map raw labels to target labels using aliases (exact or regex).
    If verify_bucket is provided (dict), record raw->target hits for verification.
    Returns: list of {id, text, labels} where labels are the *target* (mapped) labels.
    """
    alias_matchers = compile_aliases(label_aliases)
    rows_by_id = {}
    for sid, entry in raw_dict.items():
        text = (entry.get("text", "") or "").strip()
        labs_entries = entry.get("labels", [])
        mapped = set()
        for le in labs_entries:
            raw_list = le.get("label", [])
            if isinstance(raw_list, str):
                raw_list = [raw_list]
            for raw in raw_list:
                for tgt, matcher in alias_matchers:
                    if matcher(raw):
                        mapped.add(tgt)
                        if verify_bucket is not None:
                            verify_bucket[tgt][raw] += 1
        kept = [t for t in target_labels if t in mapped]
        if kept:
            rows_by_id[sid] = {"id": sid, "text": text, "labels": kept}
    return list(rows_by_id.values())

def write_alias_verification(outdir, verify_counts, family_name):
    """
    verify_counts: dict[target_label] -> Counter(raw_label -> count)
    Writes CSV and JSON; prints top matches.
    """
    rows = []
    for tgt, ctr in verify_counts.items():
        for raw, c in ctr.most_common():
            rows.append({"family": family_name, "target_label": tgt, "raw_label": raw, "count": int(c)})
    if rows:
        vcsv  = os.path.join(outdir, f"alias_verification_{family_name}.csv")
        vjson = os.path.join(outdir, f"alias_verification_{family_name}.json")
        pd.DataFrame(rows).to_csv(vcsv, index=False)
        with open(vjson, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\n[Verifier] Wrote alias verification for {family_name} →")
        print(" ", vcsv)
        print(" ", vjson)
        for tgt, ctr in verify_counts.items():
            sample = ", ".join([f"{raw}({ctr[raw]})" for raw in list(ctr.most_common(8))])
            print(f"[Verifier] {family_name}  {tgt}  ←  {sample}")
    else:
        print(f"\n[Verifier] No alias matches recorded for {family_name} "
              f"(either no aliases provided for these targets, or none matched).")

def has_aliases_for_targets(label_aliases, targets):
    return any(t in label_aliases for t in targets)

# ---------- features ----------
def build_topic_features(topics_csv, texts, max_topics=50, topk_words_per_topic=20):
    """
    Accepts either long format (topic, word) or wide format (multiple word columns).
    Returns: dict span_id -> feature dict
    """
    df = pd.read_csv(topics_csv)
    topic_to_words = defaultdict(list)
    cols = [c for c in df.columns if isinstance(c, str)]
    lower_cols = [c.lower() for c in cols]
    if (("topic" in lower_cols or "topic_id" in lower_cols) and "word" in lower_cols):
        tcol = cols[lower_cols.index("topic")] if "topic" in lower_cols else cols[lower_cols.index("topic_id")]
        wcol = cols[lower_cols.index("word")]
        for _, r in df.iterrows():
            topic_to_words[str(r[tcol])].append(str(r[wcol]).strip().lower())
    else:
        # wide format: each row = a topic
        for i, row in df.iterrows():
            words = []
            for c in cols:
                val = str(row[c]).strip()
                if val and val.lower() != "nan":
                    words.append(val.lower())
            words = [w for w in words if w]
            if words:
                topic_to_words[str(i)] = words

    # truncate & compile patterns
    def numkey(x):
        m = re.sub(r"[^0-9]", "", x or "")
        return int(m) if m else 0
    topic_ids = sorted(topic_to_words.keys(), key=numkey)[:max_topics]
    for t in topic_ids:
        topic_to_words[t] = topic_to_words[t][:topk_words_per_topic]

    topic_patterns = {
        t: re.compile(r"\b(" + "|".join([re.escape(w) for w in ws if w]) + r")\b", flags=re.IGNORECASE)
        for t, ws in topic_to_words.items() if ws
    }

    features = {}
    for sid, text in texts.items():
        feats = {}
        for t, pat in topic_patterns.items():
            matches = re.findall(pat, text)
            feats[f"topic_{t}_count"] = len(matches)
            feats[f"topic_{t}_has"] = int(len(matches) > 0)
        # simple cues (useful for Interactional)
        txt = (text or "").lower()
        feats["has_question"] = int("?" in text)
        feats["has_please"]  = int("please" in txt)
        feats["has_thanks"]  = int("thank" in txt or "thanks" in txt)
        feats["len_words"]   = len(re.findall(r"\w+", txt))
        features[sid] = feats
    return features

# ---------- sampling / splitting ----------
def greedy_round_robin_sample(examples, selected_labels, per_label=60, max_total=None, allow_multilabel=True):
    """
    Ensure up to 'per_label' examples for each label, sampling by span id (no duplicates).
    """
    by_label = {lab: [] for lab in selected_labels}
    for ex in examples:
        for lab in ex["labels"]:
            if lab in by_label:
                by_label[lab].append(ex["id"])

    picked_ids = set()
    order = sorted(selected_labels, key=lambda l: -len(by_label[l]))  # start from frequent
    counts = Counter()

    def can_pick(sid):
        return sid not in picked_ids

    something_added = True
    while something_added:
        something_added = False
        for lab in order:
            if counts[lab] >= per_label:
                continue
            for sid in by_label[lab]:
                if not can_pick(sid):
                    continue
                picked_ids.add(sid)
                labs_here = [x for x in examples_by_id[sid]["labels"] if x in selected_labels]
                for L in labs_here:
                    counts[L] += 1
                something_added = True
                break
        if max_total is not None and len(picked_ids) >= max_total:
            break

    sampled = [examples_by_id[sid] for sid in picked_ids]
    return sampled, counts

def stratified_split_by_id(sampled, dev_frac=0.15, test_frac=0.15, seed=13):
    random.Random(seed).shuffle(sampled)
    n = len(sampled)
    n_test = max(1, int(n * test_frac))
    n_dev  = max(1, int(n * dev_frac))
    test = sampled[:n_test]
    dev  = sampled[n_test:n_test+n_dev]
    train= sampled[n_test+n_dev:]
    for r in train: r["split"] = "train"
    for r in dev:   r["split"]  = "dev"
    for r in test:  r["split"]  = "test"
    return train + dev + test

def explode_to_frame(records, label_space):
    """
    Always return schema with id/text/split + label columns (even if empty).
    """
    cols = ["id", "text", "split"] + list(label_space)
    if not records:
        return pd.DataFrame(columns=cols)
    rows = []
    for r in records:
        base = {"id": r["id"], "text": r["text"], "split": r["split"]}
        for lab in label_space:
            base[lab] = int(lab in r["labels"])
        rows.append(base)
    df = pd.DataFrame(rows)
    for lab in label_space:
        if lab not in df.columns:
            df[lab] = 0
    return df[["id","text","split"] + list(label_space)]

def attach_feats(df, feats):
    """
    Safe merge: if df is empty, no-op; else left-join by id.
    """
    if df.empty:
        return df
    feat_df = (
        pd.DataFrame.from_dict(feats, orient="index")
        .reset_index().rename(columns={"index": "id"})
    )
    return df.merge(feat_df, on="id", how="left")

# ---------- raw mini JSON writers ----------
def save_raw_toy_json(out_path, source_raw_dict, sampled_records):
    """
    Write a mini JSON toyset using the *original* entries,
    plus 'selected_labels' (mapped labels used in toy task) and 'split'.
    """
    out = {}
    # Build a lookup from id -> (selected_labels, split)
    meta = {r["id"]: {"selected_labels": r.get("labels", []), "split": r.get("split", "train")} for r in sampled_records}
    for sid in meta.keys():
        if sid in source_raw_dict:
            # Copy original entry
            entry = dict(source_raw_dict[sid])
            # Attach metadata without altering original fields
            entry["selected_labels"] = meta[sid]["selected_labels"]
            entry["split"] = meta[sid]["split"]
            out[sid] = entry
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[RAW] Wrote mini JSON toyset → {out_path}  (items: {len(out)})")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("\nCopy/paste command:\n", DEFAULT_CMD, "\n")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="labels_config.json")
    ap.add_argument("--interactional_json", type=str, default="EPPC_output_json/subsentence_interactional_label.json")
    ap.add_argument("--goal_json", type=str, default="EPPC_output_json/subsentence_goal_oriented_label.json")
    ap.add_argument("--topics_csv", type=str, default="Data/topics_top20_words.csv")
    ap.add_argument("--outdir", type=str, default="toysets_out")
    ap.add_argument("--per_label", type=int, default=60)
    ap.add_argument("--dev_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load config and data
    cfg = load_json(args.config)
    inter_labels = cfg["interactional_labels"]
    goal_labels  = cfg["goal_labels"]
    label_aliases = cfg.get("label_aliases", {})

    inter_raw = load_json(args.interactional_json)
    goal_raw  = load_json(args.goal_json)

    # --- Family-aware extraction (aliases only if provided for those targets) ---
    verify_counts_inter = defaultdict(Counter)
    verify_counts_goal  = defaultdict(Counter)

    if label_aliases and has_aliases_for_targets(label_aliases, inter_labels):
        inter_examples = extract_selected_examples_with_aliases(
            inter_raw, inter_labels, label_aliases, verify_bucket=verify_counts_inter
        )
        write_alias_verification(args.outdir, verify_counts_inter, "interactional")
    else:
        inter_examples = extract_selected_examples(inter_raw, set(inter_labels))

    if label_aliases and has_aliases_for_targets(label_aliases, goal_labels):
        goal_examples = extract_selected_examples_with_aliases(
            goal_raw, goal_labels, label_aliases, verify_bucket=verify_counts_goal
        )
        write_alias_verification(args.outdir, verify_counts_goal, "goal")
    else:
        goal_examples = extract_selected_examples(goal_raw, set(goal_labels))

    print(f"\n[Counts] Interactional candidate examples: {len(inter_examples)}")
    print(f"[Counts] Goal-Oriented candidate examples: {len(goal_examples)}")
    if len(inter_examples) == 0:
        print("[WARN] No Interactional examples found. Check labels/aliases in your config.")
    if len(goal_examples) == 0:
        print("[WARN] No Goal-Oriented examples found. Check labels/aliases in your config.")

    # Global index for sampler
    global examples_by_id
    examples_by_id = {ex["id"]: ex for ex in inter_examples + goal_examples}

    # Sample per family
    inter_sampled, inter_counts = greedy_round_robin_sample(
        inter_examples, inter_labels, per_label=args.per_label, allow_multilabel=True
    )
    goal_sampled,  goal_counts  = greedy_round_robin_sample(
        goal_examples, goal_labels, per_label=args.per_label, allow_multilabel=True
    )

    # Split
    inter_split = stratified_split_by_id(inter_sampled, dev_frac=args.dev_frac, test_frac=args.test_frac)
    goal_split  = stratified_split_by_id(goal_sampled,  dev_frac=args.dev_frac,  test_frac=args.test_frac)

    # Topic features
    inter_texts = {r["id"]: r["text"] for r in inter_split}
    goal_texts  = {r["id"]: r["text"] for r in goal_split}
    topic_feats_inter = build_topic_features(args.topics_csv, inter_texts)
    topic_feats_goal  = build_topic_features(args.topics_csv, goal_texts)

    # Attach features & explode labels to wide format
    inter_df = explode_to_frame(inter_split, inter_labels)
    goal_df  = explode_to_frame(goal_split,  goal_labels)
    inter_df = attach_feats(inter_df, topic_feats_inter)
    goal_df  = attach_feats(goal_df,  topic_feats_goal)

    # Save CSVs (ML-ready)
    inter_csv = os.path.join(args.outdir, "interactional_toyset.csv")
    goal_csv  = os.path.join(args.outdir,  "goal_toyset.csv")
    inter_df.to_csv(inter_csv, index=False)
    goal_df.to_csv(goal_csv, index=False)

    # NEW: Save raw-style mini JSON toysets (for inspection / downstream tooling)
    inter_raw_out = os.path.join(args.outdir, "interactional_toyset_raw.json")
    goal_raw_out  = os.path.join(args.outdir, "goal_toyset_raw.json")
    save_raw_toy_json(inter_raw_out, inter_raw, inter_split)
    save_raw_toy_json(goal_raw_out,  goal_raw,  goal_split)

    # Summaries
    def summarize(df, labels, name):
        print(f"\n=== {name} ===")
        if df.empty:
            print("(empty)")
            return
        print(df["split"].value_counts())
        for sp in ["train", "dev", "test"]:
            sdf = df[df["split"] == sp]
            counts = {lab: int(sdf[lab].sum()) for lab in labels}
            print(sp, counts)

    summarize(inter_df, inter_labels, "Interactional")
    summarize(goal_df,  goal_labels,  "Goal-Oriented")

    # Save counts summary
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "interactional_counts": {k: int(v) for k, v in inter_counts.items()},
            "goal_counts": {k: int(v) for k, v in goal_counts.items()},
            "config": cfg
        }, f, indent=2)

    print("\nDone. Files written to:", args.outdir)
    print("\nCopy/paste command:\n", DEFAULT_CMD, "\n")
