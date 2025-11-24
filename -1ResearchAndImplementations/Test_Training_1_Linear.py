"""
Semantic Baseline (LSA) for Single-Label Classification

What it does
- Loads: subsentence_goal_oriented_label.json and sentence_goal_oriented_label.json
- Uses span text if present, otherwise uses full text
- TF-IDF (word 1-2 grams) -> TruncatedSVD (LSA) -> L2 normalize
- Appends cosine similarity to class prototypes as extra features
- Class-weighted Logistic Regression
- Stratified split with fallback if needed
- Saves metrics.json, predictions.json, label_map.json, confusion_matrix.png, per_class_f1.png

Run
  python lsa_baseline.py
Optional flags
  --subsentence /path/to/subsentence_goal_oriented_label.json
  --sentence   /path/to/sentence_goal_oriented_label.json
  --outdir     /path/to/output_root
  --seed       42
  --test_size  0.2
  --svd_dim    300

Requires
  pip install scikit-learn matplotlib numpy
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def load_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids, texts, labels = [], [], []
    skipped = 0
    for k, v in data.items():
        span = (v.get("span") or "").strip()
        text = (v.get("text") or "").strip()
        content = span if span else text

        labs = v.get("labels", [])
        if not labs or "label" not in labs[0] or not labs[0]["label"]:
            skipped += 1
            continue
        lab = labs[0]["label"][0]

        if not content or not lab:
            skipped += 1
            continue

        ids.append(k)
        texts.append(content)
        labels.append(lab)

    return ids, texts, labels, skipped


def make_label_maps(labels):
    uniq = sorted(set(labels))
    label2id = {lab: i for i, lab in enumerate(uniq)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def build_lsa(train_texts, val_texts, n_components=300, max_word_feats=80000):
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_word_feats
    )
    X_tr_tf = tfidf.fit_transform(train_texts)
    X_va_tf = tfidf.transform(val_texts)

    n_comp = max(2, min(n_components, X_tr_tf.shape[1] - 1))  # safe bound
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_tr = svd.fit_transform(X_tr_tf)
    X_va = svd.transform(X_va_tf)

    X_tr = normalize(X_tr)  # L2
    X_va = normalize(X_va)
    return X_tr, X_va, tfidf, svd


def add_prototype_similarities(X_tr, y_tr, X_va, n_classes):
    # centroids in LSA space
    centroids = np.zeros((n_classes, X_tr.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = (y_tr == c)
        if np.any(mask):
            centroids[c] = X_tr[mask].mean(axis=0)
        else:
            centroids[c] = 0.0
    centroids = normalize(centroids)

    # cosine similarity = dot for normalized vectors
    sims_tr = X_tr @ centroids.T
    sims_va = X_va @ centroids.T

    X_tr_aug = np.hstack([X_tr, sims_tr])
    X_va_aug = np.hstack([X_va, sims_va])
    return X_tr_aug, X_va_aug


def train_eval(X_tr, y_tr, X_va, y_va, n_classes):
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto"
    )
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_va)
    try:
        y_prob = clf.predict_proba(X_va)
    except Exception:
        y_prob = None

    idx_labels = list(range(n_classes))
    acc = accuracy_score(y_va, y_hat)
    macro_f1 = f1_score(y_va, y_hat, average="macro", labels=idx_labels)
    report = classification_report(y_va, y_hat, labels=idx_labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_va, y_hat, labels=idx_labels)
    return clf, y_hat, y_prob, acc, macro_f1, report, cm


def plot_confusion_matrix(cm, tick_labels, out_path: Path):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Validation)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(report_named, label_names, out_path: Path):
    f1s = [report_named.get(lab, {}).get("f1-score", 0.0) for lab in label_names]
    fig = plt.figure(figsize=(8, 10))
    y_pos = np.arange(len(label_names))
    plt.barh(y_pos, f1s)
    plt.yticks(y_pos, label_names)
    plt.xlabel("F1")
    plt.title("Per-Class F1 (Validation)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def process_file(in_path: Path, out_root: Path, test_size: float, seed: int, svd_dim: int):
    dataset = in_path.stem
    out_dir = out_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, texts, labels, skipped = load_data(in_path)
    if len(texts) < 10:
        print(f"[{dataset}] Insufficient data: usable={len(texts)} skipped={skipped}")
        return

    label2id, id2label = make_label_maps(labels)
    y_idx = np.array([label2id[l] for l in labels], dtype=int)

    # Stratified split; fallback to random if class counts are too small
    try:
        X_tr_texts, X_va_texts, y_tr, y_va, ids_tr, ids_va = train_test_split(
            texts, y_idx, ids, test_size=test_size, random_state=seed, stratify=y_idx
        )
        split_note = "stratified"
    except ValueError:
        X_tr_texts, X_va_texts, y_tr, y_va, ids_tr, ids_va = train_test_split(
            texts, y_idx, ids, test_size=test_size, random_state=seed, stratify=None
        )
        split_note = "random"

    # LSA features
    X_tr_lsa, X_va_lsa, tfidf, svd = build_lsa(X_tr_texts, X_va_texts, n_components=svd_dim)

    # Prototype similarities
    X_tr_aug, X_va_aug = add_prototype_similarities(X_tr_lsa, y_tr, X_va_lsa, n_classes=len(label2id))

    # Train and evaluate
    clf, y_hat_idx, y_prob, acc, macro_f1, report_idx, cm = train_eval(
        X_tr_aug, y_tr, X_va_aug, y_va, n_classes=len(label2id)
    )

    inv = {i: lab for lab, i in label2id.items()}
    label_names = [inv[i] for i in range(len(inv))]
    y_true_labs = [inv[i] for i in y_va]
    y_pred_labs = [inv[i] for i in y_hat_idx]

    # Convert classification_report keys to label names
    report_named = {}
    for i, lab in inv.items():
        d = report_idx.get(str(i), {})
        if not d:
            report_named[lab] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
        else:
            report_named[lab] = d

    # Save artifacts
    save_json(out_dir / "label_map.json", {"label2id": label2id, "id2label": {str(i): lab for i, lab in inv.items()}})
    save_json(out_dir / "metrics.json", {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "split": split_note,
        "per_class_f1": {lab: report_named.get(lab, {}).get("f1-score", 0.0) for lab in label_names},
        "support": {lab: report_named.get(lab, {}).get("support", 0) for lab in label_names}
    })

    preds = []
    for i, ex_id in enumerate(ids_va):
        rec = {
            "id": ex_id,
            "text": X_va_texts[i],
            "gold_label": y_true_labs[i],
            "pred_label": y_pred_labs[i],
            "correct": bool(y_true_labs[i] == y_pred_labs[i]),
        }
        if y_prob is not None:
            top = np.argsort(-y_prob[i])[:3].tolist()
            rec["topk"] = [{"label": label_names[j], "prob": float(y_prob[i][j])} for j in top]
        preds.append(rec)
    save_json(out_dir / "predictions.json", preds)

    # Plots
    plot_confusion_matrix(cm, tick_labels=[str(i) for i in range(len(inv))], out_path=out_dir / "confusion_matrix.png")
    plot_per_class_f1(report_named, label_names, out_path=out_dir / "per_class_f1.png")

    print(f"[{dataset}] usable={len(texts)} skipped={skipped} "
          f"acc={acc:.3f} macro_f1={macro_f1:.3f} split={split_note}")
    print(f"Saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsentence", type=str, default=str(Path("EPPC_output_json/subsentence_goal_oriented_label_filtered.json")))
    parser.add_argument("--sentence", type=str, default=str(Path("EPPC_output_json/sentence_goal_oriented_label_filtered.json")))
    parser.add_argument("--outdir", type=str, default=str(Path("EPPC_output_json/baseline_step2_lsa_outputs")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--svd_dim", type=int, default=300)
    args = parser.parse_args()

    subsentence_path = Path(args.subsentence)
    sentence_path = Path(args.sentence)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not subsentence_path.exists():
        print(f"Missing file: {subsentence_path}")
    else:
        process_file(subsentence_path, out_root, test_size=args.test_size, seed=args.seed, svd_dim=args.svd_dim)

    if not sentence_path.exists():
        print(f"Missing file: {sentence_path}")
    else:
        process_file(sentence_path, out_root, test_size=args.test_size, seed=args.seed, svd_dim=args.svd_dim)


if __name__ == "__main__":
    main()
