"""
Simple Tkinter Annotation Tool — pick dataset by CASE_STEM
----------------------------------------------------------
How to use:
1) Set CASE_STEM to match your split files in Data/WOVEN/split_by_case_type/
   (e.g., "Clinical_Question", "Medication", "Statement_Billing_or_Insurance_Question").
2) Run: python WOVEN_step_4_GUI_v2.py
3) Checkboxes: 0 = unchecked, 1 = checked. Use Save, Next, Previous.

Files:
- Loads from:   Data/WOVEN/split_by_case_type/{CASE_STEM}.jsonl/.json/.csv
- Saves to:     Data/WOVEN/split_by_case_type/{CASE_STEM}_annotated_output.jsonl/.csv
- Progress at:  Data/WOVEN/split_by_case_type/{CASE_STEM}_progress.json
"""

import os
import json
import hashlib
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# ======= PICK YOUR DATASET HERE =======
CASE_STEM = "Statement_Billing_or_Insurance_Question"   # <-- change this to the set you want

# ======= Paths (no filtering/normalization) =======
BASE_DIR = "Data/WOVEN/split_by_case_type"
CANDIDATES = [
    os.path.join(BASE_DIR, f"{CASE_STEM}.jsonl"),
    os.path.join(BASE_DIR, f"{CASE_STEM}.json"),
    os.path.join(BASE_DIR, f"{CASE_STEM}.csv"),
]
INPUT_FILE = next((p for p in CANDIDATES if os.path.exists(p)), CANDIDATES[0])

JSON_OUT = os.path.join(BASE_DIR, f"{CASE_STEM}_annotated_output.jsonl")
CSV_OUT  = os.path.join(BASE_DIR, f"{CASE_STEM}_annotated_output.csv")
PROGRESS_FILE = os.path.join(BASE_DIR, f"{CASE_STEM}_progress.json")

# ======= Flags in the UI =======
FLAG_COLUMNS = [
    "patient-to-provider",
    "provider-to-patient",
    "provider-to-provider",
    "is_telephone_note",
    "is_content",
    "is_multiple",
]

DEFAULTS = {
    "patient-to-provider": 0,
    "provider-to-patient": 0,
    "provider-to-provider": 1,
    "is_telephone_note": 1,
    "is_content": 1,
    "is_multiple": 1,   # default 1
}

TEXT_COL = "Pt. Case Description"

# ======= Helpers =======
def file_hash(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except FileNotFoundError:
        pass
    return h.hexdigest()

def load_dataframe(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    if ext == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported input format. Use .jsonl, .json, or .csv")

def ensure_flag_columns(df):
    for col, val in DEFAULTS.items():
        if col not in df.columns:
            df[col] = val
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val).astype(int)
    if TEXT_COL not in df.columns:
        df[TEXT_COL] = ""
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    return df

def save_outputs(df):
    tmp_json = JSON_OUT + ".tmp"
    df.to_json(tmp_json, orient="records", lines=True, force_ascii=False)
    os.replace(tmp_json, JSON_OUT)

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
        self.title(f"Annotator — {CASE_STEM}")
        self.geometry("980x640")

        self.df = df.reset_index(drop=True)
        self.n = len(self.df)
        self.progress_key = progress_key

        if self.n == 0:
            messagebox.showerror("No data", f"No rows in {INPUT_FILE}")
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

        ttk.Label(self, text=f"{TEXT_COL}:").pack(anchor="w", padx=10)
        self.text = tk.Text(self, height=18, wrap="word")
        self.text.pack(fill="both", expand=True, padx=10, pady=6)
        self.text.configure(state="disabled")

        checks = ttk.LabelFrame(self, text="Annotations")
        checks.pack(fill="x", padx=10, pady=6)
        self.vars = {name: tk.IntVar(value=DEFAULTS[name]) for name in FLAG_COLUMNS}
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

        desc = "" if pd.isna(row.get(TEXT_COL, "")) else str(row.get(TEXT_COL, ""))
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
        messagebox.showinfo("Saved", f"Saved:\n{JSON_OUT}\n{CSV_OUT}")

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
    print(f"Loading: {INPUT_FILE}")
    df = load_dataframe(INPUT_FILE)
    df = ensure_flag_columns(df)

    # progress key is tied only to the exact input file
    progress_key = file_hash(INPUT_FILE)

    app = AnnotatorApp(df, progress_key)
    try:
        app.mainloop()
    except Exception:
        pass
