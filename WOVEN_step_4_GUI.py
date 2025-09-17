"""
Tkinter Annotation Tool — JSON-first workflow
Usage:
1) Set INPUT_FILE to your JSON/JSONL file (CSV allowed but not preferred).
2) Optionally set CASE_TYPE_FILTER (e.g., "Clinical_Question"). Leave as None for all.
3) Run:  python WOVEN_step_4_GUI.py
4) Checkboxes: 0 = unchecked, 1 = checked. Use Save, Next, or Previous.

Outputs:
- Primary: annotated_output.jsonl (JSON Lines, for reliable reload)
- Secondary: annotated_output.csv (convenience export)
- Progress: progress.json (remembers last index for this file+filter)

Shortcuts: Ctrl+S = Save, Ctrl+N = Next, Ctrl+P = Previous
"""

import os
import sys
import re
import json
import hashlib
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# ======= SET YOUR INPUT FILE HERE (prefer JSON/JSONL) =======
# Examples:
# INPUT_FILE = "Clinical_Question.json"
# INPUT_FILE = "annotated_output.jsonl"
INPUT_FILE = "Data/WOVEN/split_by_case_type/Clinical_Question.json"

# ======= CASE TYPE FILTER (None to load all) =======
# Examples: None, "Clinical_Question", "Medication", "Other"
CASE_TYPE_FILTER = "Clinical_Question"

# ======= Output file names (JSON primary, CSV secondary) =======
JSON_OUT = f"Data/WOVEN/split_by_case_type/{CASE_TYPE_FILTER}_annotated_output.jsonl"
CSV_OUT = f"Data/WOVEN/split_by_case_type/{CASE_TYPE_FILTER}_annotated_output.csv"
PROGRESS_FILE = f"Data/WOVEN/split_by_case_type/{CASE_TYPE_FILTER}_progress.json"

# Checkboxes to annotate
FLAG_COLUMNS = [
    "patient-to-provider",
    "provider-to-patient",
    "provider-to-provider",
    "is_telephone_note",
    "is_content",
]

# Default values if columns are missing
DEFAULTS = {
    "patient-to-provider": 0,
    "provider-to-patient": 0,
    "provider-to-provider": 1,
    "is_telephone_note": 1,
    "is_content": 1,
}

# ======= Helpers =======
def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_case_type(val):
    """Lowercase, strip, collapse spaces to underscores. NaN -> 'unknown'."""
    if pd.isna(val):
        return "unknown"
    s = str(val).strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s

def load_dataframe(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl"):
        return pd.read_json(path, lines=True)
    elif ext == ".csv":
        print("Warning: CSV is less safe than JSON/JSONL for long text fields.")
        return pd.read_csv(path)
    else:
        raise ValueError("Unsupported input format. Use .json, .jsonl, or .csv")

def ensure_flag_columns(df):
    for col, val in DEFAULTS.items():
        if col not in df.columns:
            df[col] = val
    return df

def save_outputs(df):
    # Primary JSONL
    tmp_json = JSON_OUT + ".tmp"
    df.to_json(tmp_json, orient="records", lines=True, force_ascii=False)
    os.replace(tmp_json, JSON_OUT)
    # Secondary CSV
    tmp_csv = CSV_OUT + ".tmp"
    df.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, CSV_OUT)

def load_progress(progress_path, key):
    if not os.path.exists(progress_path):
        return 0
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("key") == key:
            return int(data.get("index", 0))
    except Exception:
        pass
    return 0

def save_progress(progress_path, key, index):
    payload = {"key": key, "index": int(index)}
    tmp = progress_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, progress_path)

# ======= Tk App =======
class AnnotatorApp(tk.Tk):
    def __init__(self, df, progress_key):
        super().__init__()
        self.title("Clinical Role Annotator (JSON-first)")
        self.geometry("980x640")

        self.df = df.reset_index(drop=True)
        self.n = len(self.df)
        self.progress_key = progress_key

        if self.n == 0:
            messagebox.showerror("No data", "No rows to annotate after filtering.")
            self.destroy()
            return

        self.index = load_progress(PROGRESS_FILE, self.progress_key)
        if self.index >= self.n:
            self.index = 0

        self.create_widgets()
        self.bind_events()
        self.load_record(self.index)

    def create_widgets(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=6)
        self.idx_label = ttk.Label(top, text="Record 0/0")
        self.idx_label.pack(side="left")

        ttk.Label(self, text="Pt. Case Description:").pack(anchor="w", padx=10)
        self.text = tk.Text(self, height=18, wrap="word")
        self.text.pack(fill="both", expand=True, padx=10, pady=6)
        self.text.configure(state="disabled")

        checks = ttk.LabelFrame(self, text="Annotations")
        checks.pack(fill="x", padx=10, pady=6)
        self.vars = {name: tk.IntVar(value=0) for name in FLAG_COLUMNS}
        for name in FLAG_COLUMNS:
            ttk.Checkbutton(checks, text=name, variable=self.vars[name]).pack(side="left", padx=6)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=10)
        ttk.Button(btns, text="Save", command=self.on_save).pack(side="left")
        ttk.Button(btns, text="◀ Previous", command=self.on_prev).pack(side="right")
        ttk.Button(btns, text="Next ▶", command=self.on_next).pack(side="right", padx=(0, 6))

    def bind_events(self):
        self.bind("<Control-s>", lambda e: self.on_save())
        self.bind("<Control-n>", lambda e: self.on_next())
        self.bind("<Control-p>", lambda e: self.on_prev())

    def load_record(self, i):
        i = max(0, min(i, self.n - 1))
        row = self.df.iloc[i]
        self.idx_label.config(text=f"Record {i+1} / {self.n}")

        desc = "" if pd.isna(row.get("Pt. Case Description", "")) else str(row.get("Pt. Case Description", ""))
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", desc)
        self.text.configure(state="disabled")

        for name in FLAG_COLUMNS:
            val = row.get(name, DEFAULTS[name])
            try:
                self.vars[name].set(int(val))
            except Exception:
                self.vars[name].set(DEFAULTS[name])

        self.index = i
        save_progress(PROGRESS_FILE, self.progress_key, self.index)

    def write_back_current_flags(self):
        for name in FLAG_COLUMNS:
            self.df.at[self.index, name] = int(self.vars[name].get())

    def on_save(self):
        if self.n == 0:
            return
        self.write_back_current_flags()
        save_outputs(self.df)
        save_progress(PROGRESS_FILE, self.progress_key, self.index)
        messagebox.showinfo("Saved", f"Saved to:\n{JSON_OUT}\n{CSV_OUT}")

    def on_next(self):
        if self.n == 0:
            return
        self.write_back_current_flags()
        save_outputs(self.df)
        next_i = (self.index + 1) % self.n
        self.load_record(next_i)

    def on_prev(self):
        if self.n == 0:
            return
        self.write_back_current_flags()
        save_outputs(self.df)
        prev_i = (self.index - 1) % self.n
        self.load_record(prev_i)

# ======= Main =======
if __name__ == "__main__":
    df = load_dataframe(INPUT_FILE)
    df = ensure_flag_columns(df)

    # Normalize and filter by Case Type (works for JSON or CSV input)
    if "Case Type" in df.columns:
        key_col = "_case_key"
        df[key_col] = df["Case Type"].apply(normalize_case_type)
        if CASE_TYPE_FILTER is not None:
            wanted = normalize_case_type(CASE_TYPE_FILTER)
            subset = df[df[key_col] == wanted].copy()
            if len(subset) == 0:
                unique_keys = sorted(df[key_col].dropna().unique())
                print(f"Requested Case Type: {CASE_TYPE_FILTER!r} -> normalized '{wanted}'")
                print("No rows matched. Available normalized case types:")
                for k in unique_keys:
                    print(f"  - {k}")
                sys.exit(1)
            print(f"Loaded only Case Type = {CASE_TYPE_FILTER} ({len(subset)} rows).")
            df = subset
        else:
            print(f"Loaded all Case Types ({len(df)} rows).")
    else:
        print('Column "Case Type" not found. Loading all rows.')

    # Progress key ties resume to this exact input + filter
    src_sig = file_hash(INPUT_FILE)
    filt_sig = normalize_case_type(CASE_TYPE_FILTER) if CASE_TYPE_FILTER else "all"
    progress_key = f"{src_sig}|{filt_sig}"

    app = AnnotatorApp(df, progress_key)
    try:
        app.mainloop()
    except Exception:
        pass
