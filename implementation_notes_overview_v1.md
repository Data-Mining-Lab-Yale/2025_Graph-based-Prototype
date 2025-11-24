# ResearchAndImplementations

This folder contains all research code and experimental pipelines developed during the postdoctoral project. It includes preprocessing scripts, text-to-graph conversion modules, graph-based models, experiment runners, and analysis utilities used for clause-level classification and related studies.

The goal of this documentation is to provide a **high-level map** of the folder and a **reusable template** for describing each script's purpose, inputs, and outputs.

---

## 1. Folder Purpose

The `ResearchAndImplementations/` directory is the main workspace for:

- Implementing and testing NLP preprocessing and clause-splitting methods.
- Building graph representations (dependency, SRL, AMR, narrative) from message texts.
- Training and evaluating graph-based models (e.g., GCN, GAT) and baseline ML models.
- Running semantic conflict analysis and ablation studies on the lab dataset.
- Performing error analysis and sanity checks on constructed graphs and predictions.

All scripts in this folder are considered **research prototypes**; many are experimental or exploratory and were written to answer specific methodological questions.

---

## 2. Recommended Script Documentation Format

Because this folder contains a large number of scripts, each file does **not** need a long description. Instead, we recommend using a short, consistent template. For each script you want to document, add an entry like this:

```md
### script_name.py
- **Category:** (e.g., Preprocessing / Graph Construction / Model Training / Evaluation / Utilities)
- **Purpose:** One–two sentences describing what the script does.
- **Inputs:** Main input files, folders, or command-line arguments.
- **Outputs:** Main output files, folders, or printed results.
- **Notes:** Optional details such as dependencies, assumptions, or status (e.g., prototype / stable / deprecated).
```

You can copy–paste this block and fill it in for each script or group of related scripts.

---

## 3. Example Entries (to be extended)

Below are example entries showing how to apply the template. You can edit the names and details to match the actual scripts.

### example_preprocessing_script.py
- **Category:** Preprocessing
- **Purpose:** Splits raw message texts into clause-level segments and exports them in a structured JSON/CSV format.
- **Inputs:** Raw message file (e.g., `Data/messages_raw.json`), configuration arguments for splitting rules.
- **Outputs:** Processed file with clause-level units (e.g., `Data/messages_clauses.json`).
- **Notes:** Used as the first step in the clause-level classification pipeline.

### example_graph_builder_script.py
- **Category:** Graph Construction
- **Purpose:** Builds a graph representation for each message or clause using dependency or SRL relations.
- **Inputs:** Preprocessed clause-level file, model configuration, and optional vocabulary files.
- **Outputs:** Serialized graphs (e.g., `*.pkl` or `*.json`) stored under `Data/Graphs/`.
- **Notes:** Required before running any GCN/GAT training scripts.

### example_gcn_training_script.py
- **Category:** Model Training
- **Purpose:** Trains a GCN model on the constructed graphs to predict clause-level labels.
- **Inputs:** Graph files, label files, and hyperparameter configuration.
- **Outputs:** Trained model checkpoints and evaluation logs.
- **Notes:** Produces the main performance numbers reported in the internal analyses.

### example_error_analysis_script.py
- **Category:** Evaluation / Analysis
- **Purpose:** Compares model predictions with gold labels and summarizes common error patterns.
- **Inputs:** Prediction file, gold-label file.
- **Outputs:** CSV or JSON with error cases, plus summary statistics.
- **Notes:** Useful for understanding which labels are confused and why.

---

## 4. How to Extend This Document

1. **Create categories** that match your actual structure (for example: `Preprocessing`, `Graph_Building`, `Models`, `Experiments`, `Error_Analysis`, `Utilities`).
2. Under each category, list the relevant scripts using the template in Section 2.
3. When new scripts are added, simply copy the template block and fill in the details.

This approach keeps the documentation maintainable even with more than 100 scripts while still giving future readers enough information to locate and reuse the most important components.

