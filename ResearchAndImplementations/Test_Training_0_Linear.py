"""
Baseline Step 1: TF-IDF + Logistic Regression on span-only text.

Inputs:
  - subsentence_goal_oriented_label.json
  - sentence_goal_oriented_label.json

Outputs for each dataset under /mnt/data/baseline_step1_outputs/<dataset_name>/ :
  - label_map.json
  - metrics.json               accuracy, macro_f1, per-class f1, support
  - predictions.json           id, text, gold_label, pred_label, correct, topk
  - confusion_matrix.png
  - per_class_f1.png
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.sparse import hstack
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------- paths ---------------
BASE_DIR = Path("EPPC_output_json")
INPUT_FILES = [
    BASE_DIR / "subsentence_goal_oriented_label_filtered.json",
    BASE_DIR / "sentence_goal_oriented_label_filtered.json",
]
OUT_ROOT = BASE_DIR / "baseline_step1_outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# --------------- helpers ---------------
def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
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

def make_label_maps(y_labels):
    uniq = sorted(set(y_labels))
    label2id = {lab: i for i, lab in enumerate(uniq)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label

def vectorize(train_texts, val_texts):
    # Word ngrams
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1,2), min_df=2, max_features=100_000
    )
    Xw_tr = word_vec.fit_transform(train_texts)
    Xw_va = word_vec.transform(val_texts)

    # Char ngrams
    char_vec = TfidfVectorizer(
        analyzer="char", ngram_range=(3,5), min_df=2, max_features=200_000
    )
    Xc_tr = char_vec.fit_transform(train_texts)
    Xc_va = char_vec.transform(val_texts)

    X_tr = hstack([Xw_tr, Xc_tr])
    X_va = hstack([Xw_va, Xc_va])
    return X_tr, X_va, word_vec, char_vec

def train_eval(X_tr, y_tr, X_va, y_va, n_classes):
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
        n_jobs=None
    )
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_va)
    try:
        y_prob = clf.predict_proba(X_va)
    except Exception:
        y_prob = None

    labels_idx = list(range(n_classes))
    acc = accuracy_score(y_va, y_hat)
    macro_f1 = f1_score(y_va, y_hat, average="macro", labels=labels_idx)
    report = classification_report(y_va, y_hat, labels=labels_idx, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_va, y_hat, labels=labels_idx)
    return clf, y_hat, y_prob, acc, macro_f1, report, cm

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_metrics(out_dir, acc, macro_f1, report_named, label2id):
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": {lab: report_named.get(lab, {}).get("f1-score", 0.0) for lab in label2id},
        "support": {lab: report_named.get(lab, {}).get("support", 0) for lab in label2id},
        "label_map": label2id
    }
    save_json(out_dir / "metrics.json", metrics)

def save_predictions(out_dir, ids_va, texts_va, y_true_labs, y_pred_labs, y_prob, label_order):
    records = []
    for i, _id in enumerate(ids_va):
        rec = {
            "id": _id,
            "text": texts_va[i],
            "gold_label": y_true_labs[i],
            "pred_label": y_pred_labs[i],
            "correct": bool(y_true_labs[i] == y_pred_labs[i])
        }
        if y_prob is not None:
            probs = y_prob[i]
            top = np.argsort(-probs)[:3].tolist()
            rec["topk"] = [{"label": label_order[j], "prob": float(probs[j])} for j in top]
        records.append(rec)
    save_json(out_dir / "predictions.json", records)

def plot_confusion_matrix(cm, labels, out_path):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Validation)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_per_class_f1(report_named, labels, out_path):
    f1s = [report_named.get(lab, {}).get("f1-score", 0.0) for lab in labels]
    fig = plt.figure(figsize=(8, 10))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, f1s)
    plt.yticks(y_pos, labels)
    plt.xlabel("F1")
    plt.title("Per-Class F1 (Validation)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# --------------- run on both datasets ---------------
for in_path in INPUT_FILES:
    if not in_path.exists():
        print(f"Missing file: {in_path}")
        continue

    dataset_name = in_path.stem
    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, texts, labels, skipped = load_dataset(in_path)
    if len(texts) < 10:
        print(f"[{dataset_name}] Insufficient data: {len(texts)} usable, {skipped} skipped")
        continue

    label2id, id2label = make_label_maps(labels)
    y_idx = np.array([label2id[l] for l in labels], dtype=int)

    X_tr, X_va, y_tr, y_va, ids_tr, ids_va, texts_tr, texts_va = train_test_split(
        texts, y_idx, ids, texts, test_size=0.2, random_state=42, stratify=y_idx
    )

    X_tr_vec, X_va_vec, word_vec, char_vec = vectorize(X_tr, X_va)

    clf, y_hat_idx, y_prob, acc, macro_f1, report_idx, cm = train_eval(
        X_tr_vec, y_tr, X_va_vec, y_va, n_classes=len(label2id)
    )

    inv = {i: lab for lab, i in label2id.items()}
    y_true_labs = [inv[i] for i in y_va]
    y_pred_labs = [inv[i] for i in y_hat_idx]
    label_order = [inv[i] for i in range(len(inv))]

    # reshape report to label names
    report_named = {}
    for i, lab in inv.items():
        d = report_idx.get(str(i), {})
        if not d:
            report_named[lab] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
        else:
            report_named[lab] = d

    # save artifacts
    save_json(out_dir / "label_map.json", {"label2id": label2id, "id2label": {str(i): lab for i, lab in inv.items()}})
    save_metrics(out_dir, acc, macro_f1, report_named, label2id)
    save_predictions(out_dir, ids_va, texts_va, y_true_labs, y_pred_labs, y_prob, label_order)

    plot_confusion_matrix(cm, labels=[str(i) for i in range(len(inv))], out_path=out_dir / "confusion_matrix.png")
    plot_per_class_f1(report_named, labels=label_order, out_path=out_dir / "per_class_f1.png")

    print(f"[{dataset_name}] usable={len(texts)} skipped={skipped} acc={acc:.3f} macro_f1={macro_f1:.3f}")
    print(f"Saved to: {out_dir}")
