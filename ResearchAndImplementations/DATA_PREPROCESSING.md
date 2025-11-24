# 📊 Annotation Preprocessing and Analysis Pipeline

This project provides a structured pipeline for processing, analyzing, and visualizing annotated communication data using a hierarchical codebook. The workflow includes transforming raw annotations, aligning them with structured code definitions, and generating interpretable outputs for downstream NLP and reasoning tasks.

---

## 🔁 Overview of Pipeline Modules

| Module | Description | Input | Output |
|--------|-------------|-------|--------|
| `ParseCodebook.py` | Parses the original codebook spreadsheet into a structured graph (nodes and links) in JSON format. | `EPPC_codebook_04.07.2025.xlsx` | `codebook_hierarchy.json` |
| `ListNodeNameWIndex.py` | Assigns hierarchical indices to codebook nodes for easier referencing. | `codebook_hierarchy.json` | `node_names_by_type_with_index.json` |
| `ProcessAnnotations.py` | Merges sequential message chunks into full messages and extracts annotations. | `EPPC_sentence_dataset_0505_merge.json` | `processed_messages_with_annotations.json` |
| `Message2SentencesAndSubs.py` | Splits each message into sentences and subsentences (clauses), linking to nearest annotation spans. | `processed_messages_with_annotations.json` | `messages_with_sentences_and_subsentences.json` |
| `EvaluationMessageAndAnnotations.py` | Evaluates alignment between annotation spans and sentence/subsentence segments. | `messages_with_sentences_and_subsentences.json`, `processed_messages_with_annotations.json` | Printed evaluation summary |
| `AnnotationAnalysis_v2.py` | Maps raw annotation labels to the codebook using fuzzy matching, and summarizes their frequency. | `codebook_hierarchy.json`, `processed_messages_with_annotations.json` | `annotation_code_mapping_detailed_corrected.json`, `annotation_code_frequency_summary_corrected.csv` |
| `CheckCoverage.py` | Verifies which codebook entries are used in annotations and calculates usage coverage. | `codebook_hierarchy.json`, `annotation_code_frequency_summary_corrected.csv` | `codebook_coverage_with_frequencies.csv` |
| `FrequencyVisualization.py` | Generates bar charts for code, subcode, and subsubcode usage frequency. | `annotation_code_frequency_summary_corrected.csv` | `top20_mixed_by_corrected_levels.png`, `top20_codes_only_corrected.png`, `top20_subcodes_only_corrected.png`, `top20_subsubcodes_only_corrected.png` |
| `GenerateSubTreeVisualizations.py` | Draws and saves subtree visualizations for each root node (code) using hierarchy layout. | `codebook_hierarchy.json`, `node_names_by_type_with_index.json` | `sub_tree_images/` |
| `InteractiveQueryAndVisualize_EPPC.py` | Interactive command-line tool to explore and visualize the codebook graph on demand. | `codebook_hierarchy.json` | Real-time visualizations (shown via `matplotlib`) |

---

## 📁 Outputs Summary

| File | Description |
|------|-------------|
| `codebook_hierarchy.json` | Graph-structured representation of the full codebook (nodes + links). |
| `node_names_by_type_with_index.json` | Indexed list of codes/subcodes/subsubcodes by hierarchy. |
| `processed_messages_with_annotations.json` | Reconstructed messages with attached annotations. |
| `messages_with_sentences_and_subsentences.json` | Sentence and clause (subsentence) segmentation with annotation alignment. |
| `annotation_code_mapping_detailed_corrected.json` | Detailed mapping of annotation labels to codebook entries. |
| `annotation_code_frequency_summary_corrected.csv` | Frequency table of mapped annotation codes and their levels. |
| `codebook_coverage_with_frequencies.csv` | Coverage table indicating codebook entries with and without annotations. |
| `*.png` in output folder | Frequency bar charts and subtree hierarchy diagrams. |

---

## 🎯 Purpose

This preprocessing pipeline was designed to:
- Normalize and structure annotation data
- Align annotations with hierarchical domain knowledge (via codebooks)
- Evaluate annotation span coverage
- Prepare structured, interpretable data for downstream reasoning tasks such as classification, summarization, and communication modeling

---

## 🔧 Customization

- Threshold for fuzzy matching (default: 0.6) can be tuned in `AnnotationAnalysis_v2.py`
- Tree visualizations can be regenerated for new hierarchies by updating the codebook Excel input

---

## 📌 Next Steps

The structured output (e.g., sentence/subsentence-aligned JSON, mapped codebook annotations) enables:
- Training span-based classifiers
- Narrative graph modeling
- Pattern mining and dialogue act extraction
- Storyline or communication intent visualization

---
