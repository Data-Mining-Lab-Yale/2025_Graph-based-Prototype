# Graph-Based Annotation Reasoning Prototype

This repository contains the prototype implementation of a graph-based semantic reasoning framework for **sub-sentential annotation** of unstructured patient-provider messages. It aims to support automated sub-CODE prediction through symbolic and neural reasoning over graph structures built from clause-level message segments.

## 🧠 Project Summary

In many healthcare NLP settings, patient-provider messages are annotated with semantic labels (e.g., communication intents or patient values). However, a single sentence may include multiple distinct intents or concepts that require **fine-grained sub-sentence modeling**.  
This prototype builds a graph representation from **sub-sentential units** (e.g., clauses or segments) and explores rule-based and neural reasoning for sub-CODE classification using sentence embeddings, speaker roles, and contextual structure.

## 📁 Repository Structure

graph-annotation-prototype/

├── data/ # Sample messages and label schemas (anonymized or simulated)

├── graph_builder/ # Graph construction scripts from segmented sentence units

├── models/ # Baseline classifiers and GNN prototypes

├── experiments/ # Notebooks and evaluation scripts

├── segmenter/ # Clause/sub-sentence segmentation tools (e.g., benepar)

├── utils/ # Helper functions (e.g., text preprocessing, entity extraction)

└── README.md # Project overview

text_to_graph/
 ├── base.py                # GraphBuilder base class

 ├── dependency.py          # Syntax abstraction
 
 ├── srl.py                 # Semantic abstraction (next)
 
 └── visualize.py           # Graph visualization (robust to label/text)
  
scripts/

 ├── run_one_example.py     # Debug single graph

 └── generate_all_graphs.py # Batch over dataset


Data/

 ├── subsentence_subcode_labels.json

 └── sentence_subcode_labels.json

outputs/
 ├── text2graphs_order_json/

 └── text2graphs_order_viz/

run_pipeline.py   # Main script to load text, build, and save graphs




## 🚀 Getting Started

1. **Clone the repo**:
   ```bash
   git clone https://github.com/RimiChen/graph-annotation-prototype.git
   cd graph-annotation-prototype
Install dependencies (recommended: use a virtual environment):

```CodeBlock
pip install -r requirements.txt
```

Run the graph construction script:

```CodeBlock
python graph_builder/build_graph.py --input data/sample_messages.json
```

Try the baseline rule-based sub-CODE prediction:

```CodeBlock
python experiments/rule_based_predictor.py
```
🧪 Features
Graph construction from sub-sentential segments (e.g., clauses) within clinical messages

Encodes sentence structure, speaker role, and semantic similarity

Supports rule-based and neural sub-CODE classification

Designed for early-stage prototype evaluation on small annotated subsets


📄 Citation
If you use this codebase in your research, please cite the associated paper (to appear):

```bibTex
@inproceedings{chen2025graphreasoning,
  title     = {Graph-Based Semantic Reasoning for Sub-Sentential Annotation in Patient-Provider Communication},
  author    = {Chen, Yi-Chun and [Others]},
  booktitle = {To appear},
  year      = {2025},
  url       = {https://github.com/RimiChen/graph-annotation-prototype}
}
(BibTeX will be updated once the paper is officially published.)
```

📬 Contact
For questions or contributions, please contact:

Yi-Chun (Rimi) Chen
📧 ychen74@alumni.ncsu.edu


📜 License

This code is released under the MIT License. See LICENSE for details.






