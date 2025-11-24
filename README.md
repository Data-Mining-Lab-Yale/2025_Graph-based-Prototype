# 2025 Postdoctoral Transition Repository: Graph-Based Prototype & Deliverables Archive

This repository serves two purposes:

1. **Primary (Current)** — A centralized archive documenting all **research deliverables, scripts, analyses, reports, and notes** completed during my postdoctoral appointment (May–November 2025). It supports transparent transition, reproducibility, and continuity for future team members.

2. **Historical (Original Scope)** — The repository initially hosted the prototype implementation for a **graph-based clause-level reasoning model** for provider–patient communication. These research components are preserved as a standalone subproject within the repository.

This README provides an overview of the repository structure, core components, and transition-related materials.

---

## 1. Repository Structure

```
2025_Graph-based-Prototype/
│
├── Data/
│   └── Processed datasets (EPPC, PV, Bethesda subsets, WOVEN samples)
│
├── Models/
│   └── Imported models used for experiments
│
├── ResearchAndImplementations/
│   └── Scripts and results:
│       • clause segmentation and preprocessing
│       • text-to-graph conversion (dependency, SRL, weighted variants)
│       • GNN models (GCN, GAT)
│       • semantic conflict analysis (TF-IDF, LSA, SBERT, Jaccard)
│       • decision-rule experiments
│       • auto-labeling scripts
│
├── AnalyticalNotesAndWritingAndPresentations/
│   └── Notes, diagrams, proposal drafts, literature reviews, slides
│
├── transition_list.md
│
├── Transition_plan_and_self_evaluations.md
│
├── srl_env.yaml
│
└── README.md
```

---

## 2. Current Purpose of the Repository

This repository functions as the **official transition archive** summarizing all research and implementation work since May 2025. It consolidates:

* scripts and pipelines
* processed datasets
* analysis reports
* conceptual notes
* writing deliverables
* environment files
* transition plans

The month-by-month record of tasks appears in `transition_list.md`, and grouped categories plus a self-evaluation appear in `Transition_plan_and_self_evaluations.md`.

---

## 3. Original Project Summary (Graph-Based Clause-Level Prototype)

Originally, the repository contained a prototype system for **graph-based semantic reasoning** applied to sub-sentential annotation of patient–provider messages.

### Core Research Problem

Clinical messages often contain multiple intents in a single sentence. The prototype explored:

* clause segmentation
* symbolic graph abstraction
* graph neural reasoning
* rule-based decision structures
* semantic conflict analysis

### Features

* Dependency-based, SRL-based, and weighted graph builders
* Weighted edges encoding semantic relations
* GCN/GAT classification baselines
* Conflict analysis (TF-IDF, LSA, SBERT, Jaccard)
* Local LLM labeling interface
* Integration with data quality studies

---

## 4. Transition Deliverables (May–Nov 2025)

A full list with timestamps is in `transition_list.md`. Major categories:

### A. Research & Implementation

* Graph construction modules and GNN models
* Data preprocessing pipelines
* Semantic conflict analysis toolkit
* Decision-rule and uncertainty aggregation experiments
* WOVEN and Bethesda sample set construction
* Local LLM labeling interface

### B. Writing & Presentations

* Workshop paper draft
* Two Yale AI Seed Grant proposal drafts
* Poster for Yale Postdoc Symposium
* Literature reviews: graph NLP, intent modeling, healthcare communication, virtual patients

### C. Analytical Notes

* Design documents and diagrams
* Notes on noisy-label literature
* Intent-type conceptualization

### D. Transition Preparation

* Consolidated deliverables
* Data/script cleanup
* Transition plan and summary
* Additional literature review

---

## 5. Getting Started (Graph Prototype)

### Clone the repository

```bash
git clone https://github.com/RimiChen/2025_Graph-based-Prototype.git
```

### Setup environment

```bash
conda env create -f srl_env.yaml
conda activate srl_env
```

### Example usage

```bash
python ResearchAndImplementations/graph_builder/build_graph.py --input Data/sample_messages.json
```

---

## 6. License

Released under the MIT License.

---

## 7. Contact

Yi-Chun (Rimi) Chen
Email: [ychen74@alumni.ncsu.edu](mailto:ychen74@alumni.ncsu.edu)
