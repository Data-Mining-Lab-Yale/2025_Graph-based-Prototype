# ResearchAndImplementations Script Index

This document provides a structured, category-based index of scripts in the `ResearchAndImplementations/` folder.  
It follows a table format similar to the GameTileNet `SCRIPT_INDEX`, with concise descriptions for each script.

The folder is organized into major categories reflecting the research pipeline:

- **1. Basic Processing Scripts** (message reconstruction, segmentation, cleaning, annotation analysis)
- **2. Statistical / Dataset Analysis Scripts**
- **3. Ambiguity & Representation Analysis Scripts**
- **4. Graph Label Normalization & Error Standardization**
- **5. Span Consistency & Sentence/Annotation Structure Evaluation**
- **6. Documentation & Supporting Notes**
- **7. Decision-Structure Parsing & Phrase-Chunking Tools**
- **8. Phrase Lookup & Scoring Engine**
- **9. Graph-based Modeling**
- **10. Visualization Tools**
- **11. File Conversion Tools**

Each table uses the following columns:

- **Script** – filename  
- **Main Functions** – core operations performed  
- **Goal** – the research or processing objective  
- **Key Outputs** – files or results produced  

---

## 1. Basic Processing Scripts

Scripts that convert datasets, reconstruct messages, split sentences/subsentences, or clean/inspect annotations.

| Script | Main Functions | Goal | Key Outputs |
|--------|----------------|------|-------------|
| **Basic_Processing_0_Original2Processed.py** | Converts Bethesda-style JSONL data into a normalized processed format with message IDs and annotation objects. | Standardize the raw dataset for downstream segmentation and labeling. | `Bethesda_processed_messages_with_annotations.json` |
| **Basic_Processing_0_Message2SentencesAndSubs.py** | Baseline segmentation of raw messages into sentences and subsentences using simple rules. | Provide a first-pass clause segmentation baseline. | JSON with per-sentence and per-subsentence structures. |
| **Basic_Processing_1_EvaluationMessageAndAnnotations.py** | Matches annotation spans to segmented units and computes coverage statistics. | Evaluate segmentation quality against original annotation spans. | Console statistics report. |
| **Basic_Processing_2_Message2SentencesAndSubs.py** | Improved segmentation pipeline that extracts sentences and clause-like segments with refined heuristics. | Enhance coverage and quality of clause-level units for modeling. | JSON with message → sentence → subsentence entries. |
| **Basic_Processing_3_ProcessAnnotations.py** | Cleans, consolidates, and restructures annotation definitions and fields. | Normalize annotation schemas for consistent downstream use. | Cleaned annotation JSON. |
| **Basic_Processing_4_AnnotationAnalysis.py** | Summarizes annotation types, span counts, and general properties. | Understand the structure and coverage of annotations in the dataset. | Annotation statistics JSON/CSV. |
| **Basic_Processing_5_FrequencyVisualization.py** | Generates frequency visualizations of annotation codes (e.g., top-N plots). | Inspect distribution of annotation codes and identify dominant labels. | Frequency plots such as `topN_*_corrected.png`. |
| **Clean_Data_1_filter.py** | Filters raw or processed annotations based on rules (e.g., drop invalid or unwanted entries). | Remove irrelevant or malformed annotations. | Filtered annotation JSON. |
| **Clean_Data_2_Cut_Span.py** | Shortens or trims overly long annotation spans in processed data. | Improve granularity and consistency of annotation spans. | Span-trimmed annotation JSON. |
| **Clean_Data_2_Cut_Span_non_processed_annotations.py** | Applies span trimming to non-processed/raw annotations. | Normalize span length in original annotation files. | Trimmed non-processed annotation JSON. |
| **Bethesda_Message2SentencesAndSubs_v3ud.py** | Uses spaCy UD parse cues (`advcl`, `ccomp`, `conj`, etc.) to segment text into high-quality subsentences, with length control and merging heuristics. | Produce linguistically grounded clause segmentation for the Bethesda dataset. | `Bethesda_messages_with_sentences_and_subsentences.json` |
| **EPPC_Message2SentencesAndSubs_v1.py** | NLTK + spaCy hybrid segmentation pipeline for the EPPC dataset. | Create clause-level segmentation for EPPC messages. | `messages_with_sentences_and_subsentences.json` |
| **EPPC_MessageReconstruction.py** | Reconstructs multi-turn message threads into the “final message” text using prev/next fields. | Recover accurate message units for downstream analysis and modeling. | `processed_messages_with_annotations.json` |
| **Message2SentencesAndSubs_v2.py** | Uses NLTK sentence tokenization and spaCy-based clause segmentation plus fuzzy string matching to split messages into sentences and subsentences and align them with nearby annotation texts. | Build a refined sentence + subsentence segmentation with approximate annotation-span alignment for clause-level modeling. | `EPPC_output_json/CleanedData/messages_with_sentences_and_subsentences_v2.json` |
| **Message2SentencesAndSubs_v3.py** | Segments messages into sentences and UD-based clause-like subsentences with length constraints, merges very short spans, and fuzzy-matches each segment to the closest annotation span. | Produce linguistically grounded, length-controlled clause segments for each message to improve clause-level training units. | `EPPC_output_json/CleanedData/messages_with_sentences_and_subsentences_v3ud.json` |
| **Optimize_code_structure_training_data.py** | Reads sentence/subsentence structures and separate interactional/goal annotation files, matches spans by message ID and text, and splits them into four aligned datasets by unit type and label type. | Construct sentence- and subsentence-level training JSONs with attached interactional and goal-oriented labels for modeling. | `EPPC_output_json/sentence_interactional_label.json`, `subsentence_interactional_label.json`, `sentence_goal_oriented_label.json`, `subsentence_goal_oriented_label.json` |
| **Preparation_3_per-message_structures.py** | Converts messages into a hierarchical structure (message → sentences → subsentences) with IDs for visualization. | Produce hierarchical message structures for graph-based analysis. | `messages_with_sentences_and_subsentences.json` |
| **Preparation_3_per-message_structures_v2.py** | Updated version with refined structure fields and improved per-message grouping. | Generate cleaned per-message hierarchical structures for visualization and modeling. | `messages_with_sentences_and_subsentences.json` |
| **Step1_FilterTrainingInput.py** | Loads processed messages with annotations and removes items missing span/text consistency or mapping information. | Clean input dataset before mapping annotations to units. | Filtered JSON under `EPPC_output_json/` |
| **Step2_MappingAnnotations.py** | Maps cleaned annotation spans to sentence/subsentence units using index maps and codebook metadata. | Attach structured labels to units for downstream splitting. | `sentence_label_structured.json`, `subsentence_label_structured.json` |
| **Step3_FilterOutCodes.py** | Splits structured labels into code / subcode / subsubcode per unit type and writes separate JSON files. | Produce hierarchical label splits for sentence/subsentence units. | `{sentence|subsentence}_{code|subcode|subsubcode}_labels.json` |
| **message_index_mapping_v2.py** | Rebuilds message/sentence/subsentence index maps directly from hierarchical message structure using message_id, sentence_id, subsentence_id. | Updated mapping extraction aligned with v3 segmentation. | Six index mapping and annotation-only JSON files under `EPPC_output_json/` |
| **Step2_MappingAnnotations_v2.py** | Loads cleaned annotations, sentence/subsentence span maps, codebook IDs; attaches structured labels including code_id and level. | Final structured label mapping for modeling. | `sentence_label_structured.json`, `subsentence_label_structured.json` |
| **Step4_CountAnnotations.py** | Loads structured sentence/subsentence labels and counts how many CODE/SUBCODE/SUBSUBCODE labels exist at each level. | Quick statistics on annotation label distributions across levels. | Console output (counts per level) |

---

## 2. Statistical / Dataset Analysis Scripts

Scripts computing class-level scores, distribution statistics, imbalance metrics, and dataset summaries.

| Script | Main Functions | Goal | Key Outputs |
|--------|----------------|------|-------------|
| **Analysis_1_pre_class.py** | Computes per-class F1/precision/recall from prediction logs and generates bar plots. | Evaluate model performance at the class level. | `per_class_f1.csv`, `per_class_f1_plot.png`. |
| **Analysis_1_pre_class_v2.py** | Extended version of `Analysis_1_pre_class.py` with additional metrics or refined outputs. | Provide more detailed per-class evaluation and diagnostics. | Updated metrics CSV + plots. |
| **Basic_Analysis_1_balance.py** | Computes class frequency distributions across the dataset. | Check label imbalance and skew. | Distribution tables/plots. |
| **Basic_Analysis_2_ambiguous.py** | Detects ambiguous or duplicated annotation spans and aggregates counts. | Identify potential noise sources in annotation data. | Ambiguity statistics JSON/CSV. |
| **Analysis_Processing_1_Distributions_PV.py** | Computes distribution statistics specific to the PV dataset. | Characterize the PV dataset (labels, counts, splits). | Distribution metrics and tables. |
| **Dataset_Stat_1_original.py** | Computes baseline label statistics (code, subcode, combined) for the original dataset. | Establish baseline label distributions. | Metrics CSV/JSON, optional plots. |
| **Dataset_Stat_2_with_split.py** | Computes metrics per split (e.g., Interactional vs Goal-Oriented) and generates plots. | Compare distributions across major dataset splits. | `<type>_code_counts.csv`, metrics, plots. |
| **Dataset_Stat_2_with_split_v2.py** | Refined version of `Dataset_Stat_2_with_split.py` with improved handling or formatting. | Produce cleaner split-level statistics. | Updated CSVs and plots. |
| **Dataset_Stat_3_Combined_pre_label_together.py** | Combines per-label disagreement CSVs across multiple methods into a single summary. | Summarize cross-method label disagreement patterns. | Combined disagreement CSV. |
| **Dataset_Stat_3_Combined_label_examples.py** | Merges example-level files for disagreement/conflict samples across methods. | Inspect and compare example-level disagreements. | Combined examples CSV. |
| **Dataset_Stat_3_Combined_label_pairs_together.py** | Merges ambiguous label-pair CSVs from different methods into one view. | Compare ambiguous label pairs across methods. | Combined label-pair CSV. |
| **CheckCoverage.py** | Merges the codebook hierarchy with frequency summaries to compute coverage for each label. | Identify covered, uncovered, and rarely used codes. | `codebook_coverage_with_frequencies.csv`. |
| **Data_1_map_labels_for_graph.py** | Maps labels to codebook IDs suitable for graph construction (e.g., node labels). | Standardize labels used in graph datasets. | Graph-label mapping JSON/CSV. |
| **Data_2_map_labels_for_error.py** | Maps raw error labels to normalized codebook labels for error analysis. | Harmonize error logs for downstream analysis. | Standardized error JSON/CSV. |
| **Data_2_map_labels_for_error_narrrative.py** | Same as above but for narrative-model error logs. | Normalize narrative-model errors and labels. | Standardized narrative error JSON/CSV. |
| **Data_3_map_labels_for_error_flat.py** | Flattens and standardizes error files across runs into a uniform structure. | Create a consistent error format for analysis scripts. | `<model>_errors_standardized.json` or similar. |
| **Data_4_map_labels_check.py** | Debug script to verify label mappings and check for mismatches. | Validate mapping correctness and catch inconsistencies. | Mapping check report (printed or CSV). |
| **Dataset_1_test_CLINC150.py** | Loads the CLINC150 dataset and prints out example intents and text. | Inspect an external intent dataset for comparison or intuition. | Example outputs printed to console. |
| **Dataset_1_test_DailyDialog.py** | Loads the DailyDialog dataset and prints out example dialogues with labels. | Inspect an external dialogue/intent dataset. | Example outputs printed to console. |
| **Experiments_1_MLP.py** | Trains and evaluates a simple MLP classifier on clause- or sentence-level embeddings | Provide a lightweight non-graph baseline for comparison | Accuracy/F1 metrics, predictions, logs |
| **Experiments_1_MLP_v2.py** | Improved MLP baseline with structured error logs and ID mapping | Provide more stable baseline for clause-level classification | Metrics, plots, structured `errors.json` |
| **Experiments_1_MLP_v3.py** | Further-refined MLP with better monitoring and text→ID mapping | Improve robustness of MLP baseline | Metrics, logs, `errors.json` |
| **Experiments_1_MLP_v4.py** | Final MLP variant: improved early stopping, logging, and plotting | Produce high-quality baseline for comparison with GCNs | Metrics, training curves, error logs |
| **Experiments_3_SRL_GCN_error copy.py** | Variant of SRL-GCN training script focusing on error extraction | Debug SRL-GCN behavior and collect mispredictions | `errors.json`, console logs|
| **Graph_Method_1_metric_of_expressitivity.py** | Loads a folder of JSON graphs, builds NetworkX MultiDiGraphs, converts to simple graphs, and computes structural metrics (density, degree, clustering, entropy, spectral features, WL hash). | Quantify graph complexity and expressivity across a collection of narrative/semantic graphs for comparative analysis. | `graph_metrics_{FOLDER}.json`, `graph_metrics_summary_{FOLDER}.json`, `graph_metrics_{FOLDER}.csv` |
| **Processed_0_DEP_features.py** | Loads dependency graphs, builds token-level POS features and edge indices, wraps them as PyG `Data` objects, and saves the full dataset. | Generate dependency-based graph features for ML experiments. | `dep_graph_features.pt` |
| **Processed_0_DEP_features_v2.py** | Builds global POS vocab, converts each filtered dependency graph into PyG `Data`, logs failures, and saves the compiled dataset. | Create normalized POS-feature dependency graphs for model training. | `dep_graph_features.pt`, `dep_graph_fail_log.txt` |
| **Processed_1_SRL_anchored_features.py** | Loads anchored SRL graphs, encodes node labels with BERT, builds edge indices and edge weights, and saves PyG `Data` objects. | Generate anchored SRL graph features for GCN experiments. | `processed_graph_features_anchored.pt` |
| **Processed_1_SRL_predicate_features.py** | Processes predicate-centric SRL graphs, encodes node labels with BERT, builds predicate-based edges, and exports PyG graphs. | Create predicate-centric SRL graph features for classification. | `processed_graph_features_predicate.pt` |
| **Processed_1_SRL_weighted_features.py** | Converts weighted SRL graphs into PyG format, using BERT node embeddings and numeric edge weights. | Produce edge-weighted SRL graph features for weighted GCN models. | `processed_srl_weighted_features.pt` (or `processed_graph_features.pt`, depending on run config) |
| **Processed_2_AMR_features.py** | Reads filtered AMR graphs, encodes node labels with BERT, builds edges, and saves AMR-based PyG graphs. | Generate AMR graph features as an alternative semantic representation. | `processed_graph_features_amr.pt` |
| **Processed_5_Narrative_feature.py** | Loads narrative ego-graphs, looks up AMR features for each node, builds weighted node vectors and edge attributes, and saves centered PyG graphs. | Build narrative ego-graph features aligned with AMR representations for clause-level GCN experiments. | `processed_graph_features_narrative_ego.pt` |
| **Reduced_Sample_1_summarize_alias_contributions.py** | Summarizes alias→target mappings for toysets; counts alias usage and outputs CSV/JSON + plots. | Analyze alias distributions inside toysets. | `<prefix>_alias_summary.csv`, `<prefix>_subcode_breakdown.csv`, `<prefix>_alias_summary.json`, `<prefix>_alias_totals.png` |
| **ResultAnalysis_1_MLP.py** | Loads MLP training logs and builds a comparison table of sentence vs subsentence accuracy/F1. | Summarize MLP baseline performance. | `mlp_comparison_summary.csv` |
| **Reduced_Sample_1.py** | Builds toysets with alias mapping, topic features, wide-format CSVs, raw JSONs, and split statistics. | Produce ML-ready toysets and summaries. | `interactional_toyset.csv`, `goal_toyset.csv`, raw toysets, `summary.json` |
| **Reduced_Sample_1_analysis.py** | Analyzes toyset label distributions; generates CSV summaries and comparison plots. | Visual profiling of toyset composition and splits. | CSV + PNG files under `toyset_analysis_out/` |
| **ResultAnalysis_2_graph.py** | Loads *_train_log.json files for multiple models, extracts highest F1 epoch, and builds an overall comparison summary. | Compare best F1 performance across all graph-based and baseline models. | `comparison_summary_best_f1.csv` |
| **ResultAnalysis_2_graph_v2.py** | Improved version scanning multiple result folders, detecting logs automatically, and exporting sorted comparison tables. | Comprehensive performance comparison across all model types. | `comparison_summary_best_f1.csv` |
| **ResultAnalysis_3_error_cross_models_bar.py** | Collects *_errors.json across all models and plots class-level misclassifications using grouped bar charts. | Visualize per-class error frequency across models. | `error_analysis_across_models.png`, `error_analysis_counts.json` |
| **ResultAnalysis_3_error_cross_models_heat.py** | Builds a true-label error heatmap using raw error files and codebook mapping; saves matrix + visualization. | Heatmap of true-label error counts across models. | `true_label_error_count_matrix.json`, `error_analysis_heatmap.png` |
| **ResultAnalysis_3_error_cross_models_heat_v2.py** | Extended heatmap with gold-count normalization, standardized labels, and improved formatting. | Enhanced cross-model error analysis with totals and normalization. | `error_counts_with_totals.csv`, `error_analysis_heatmap_with_totals.png` |
| **ResultAnalysis_3_error_stlye_align.py** | Standardizes error files by mapping raw true/pred labels to codebook labels; writes aligned error JSONs. | Produce normalized error datasets for cross-model comparison. | Standardized error files under `formatted_errors/` |
| **ResultAnalysis_2_graph.py** | Loads *_train_log.json for multiple models and generates a comparison table of best F1 performance. | Model comparison across all GCN/MLP graph variants. | `comparison_summary_best_f1.csv` |
| **ResultAnalysis_2_graph_v2.py** | Improved scanning over result folders, auto-detecting logs, and exporting sorted F1 tables. | Automated multi-model performance summary. | `comparison_summary_best_f1.csv` |
| **ResultAnalysis_3_error_cross_models_bar.py** | Aggregates true-label errors across models and plots horizontal bar charts. | Visualize class-level misclassification across models. | `error_analysis_across_models.png`, JSON counts |
| **ResultAnalysis_3_error_cross_models_heat.py** | Builds heatmap of true-label errors using mapping-corrected labels. | Heatmap of model error patterns by gold label. | `true_label_error_count_matrix.json`, PNG heatmap |
| **ResultAnalysis_3_error_cross_models_heat_v2.py** | Adds per-label totals, normalized counts, and improved labeling for cleaner heatmap output. | Enhanced cross-model error analysis with totals. | `error_counts_with_totals.csv`, improved heatmap PNG |
| **ResultAnalysis_3_error_stlye_align.py** | Standardizes *_errors.json by mapping raw true/pred labels to codebook labels; outputs cleaned sets. | Normalize error logs for fair cross-model comparison. | Standardized error JSONs under `formatted_errors/` |


---

## 3. Ambiguity & Representation Analysis Scripts

Scripts comparing textual representations (SBERT, TF-IDF, LSA, Jaccard), merging conflict samples, and analyzing label ambiguity.

| Script | Main Functions | Goal | Key Outputs |
|--------|----------------|------|-------------|
| **Analysis_AmbiguityCompare_MultiRep.py** | Builds multiple representations (e.g., SBERT, TF-IDF, LSA, Jaccard), finds nearest neighbors, and computes cross-label disagreements. | Analyze how different representations expose ambiguity and disagreement patterns. | `merged_all_conflict_examples.csv`, `merged_all_label_pairs.csv`, summary JSON, figures. |
| **combine_per_label_disagreement.py** | Merges per-label disagreement CSVs across different methods or configurations. | Compare disagreement statistics across methods. | `compare_*_per_label_disagreement.csv`. |
| **combine_nn_conflict_samples.py** | Merges neighbor-level conflict sample files; computes overlaps, counts, and top-K lists. | Compare and summarize nearest-neighbor conflict examples. | Combined CSVs (e.g., LONG, TOPK, OVERLAP, COUNTS). |
| **combine_ambiguous_label_pairs.py** | Combines ambiguous-label pair CSVs from multiple methods into a detailed comparison file. | Inspect ambiguous label pairs and their frequencies across methods. | `compare_*_ambiguous_label_pairs_detailed.csv`. |
| **Dataset_Stat_3_LabelAmbiguity.py** | Computes label ambiguity statistics across the dataset (e.g., confusion patterns or conflict counts). | Quantify label ambiguity at the dataset level. | Ambiguity statistics CSV/JSON. |
| **Dataset_Stat_3_LabelAmbiguity_code.py** | Variant of label ambiguity analysis at the code-only level. | Focus ambiguity statistics on high-level codes. | Code-level ambiguity CSV/JSON. |
| **Dataset_Stat_3_LabelAmbiguity_combined.py** | Computes ambiguity using combined CODE||SUBCODE labels. | Analyze ambiguity when treating code+subcode as a single category. | Combined-level ambiguity CSV/JSON. |
| **Dataset_Stat_3_LabelAmbiguity_combined_jaccard.py** | Label ambiguity analysis using Jaccard similarity on lexical features. | Assess label ambiguity using lexical overlap. | Jaccard-based ambiguity CSV/JSON. |
| **Dataset_Stat_3_LabelAmbiguity_combined_semantic_var.py** | Label ambiguity analysis using semantic variance (e.g., embedding-based measures). | Assess label ambiguity with semantic similarity metrics. | Semantic-variance ambiguity CSV/JSON. |
| **Dataset_Stat_3_LabelAmbiguity_vis.py** | Builds plots/figures for label ambiguity statistics. | Visualize label ambiguity patterns. | Ambiguity figures (PNG, etc.). |
| **Dataset_Stat_3_LabelAmbiguity_vis_correction.py** | Corrected visualization script for label ambiguity to fix or refine plotting logic. | Produce cleaner, corrected ambiguity figures. | Updated ambiguity figures. |
| **Dataset_Stat_3_LabelAmbiguity_vis_json.py** | Generates JSON-based visual summaries or configuration for ambiguity plots. | Support JSON-driven visualization or downstream tools. | JSON summary files for ambiguity. |
| **Dataset_Stat_3_AmbiguityPairs_VerticalBar.py** | Creates vertical bar charts for top ambiguous label pairs. | Visualize the most frequent ambiguous label pairs. | `ambiguous_pairs_bar_vertical.png` or similar. |
| **Dataset_Stat_3_NN_Conflicts_ToJSON.py** | Converts nearest-neighbor conflict samples from CSV to JSON (flat list + grouped by label pair). | Make NN conflict samples easier to consume in analysis or visualization. | `nn_conflicts_examples.json`, `nn_conflicts_by_pair.json`. |
| **Decision_Structure_1_analysis_pharses.py** | Analyzes frequent phrases or patterns used in decision structures or rules. | Explore phrase-level structure relevant to decision rules or labeling. | Phrase-level statistics or summaries (printed/CSV). |
| **SplitAnnotations_CrossClause.py** | Splits annotations based on clause boundaries and identifies spans that cross clauses. | Detect structurally ambiguous or multi-clause spans. | `only_cross_clause.json`, `no_cross_clause.json`. |
| **SplitAnnotations_CrossClause_Flags.py** | Extends cross-clause analysis with span flags (e.g., length, clause count). | Provide detailed diagnostics for ambiguous spans. | Span-flag JSON files. |
| **Decision_Structure_1_analysis_pharses.py** | Analyzes phrases extracted from clauses; prints phrase-level structural diagnostics | Inspect structural features of phrases before scoring | Console analysis output |
| **ParseCodebook.py** | Reads EPPC codebook Excel, parses CODE / SUB-CODE / SUB-SUB definitions, normalizes IDs, and builds node–link hierarchical JSON. | Construct full hierarchical codebook graph from Excel source. | `codebook_hierarchy.json` |
| **SentenceIndexMapping.py** | Builds message/sentence/subsentence ID → content/span maps and filtered versions containing only annotated units. | Create index-based lookup tables for all unit levels. | `message_index_mapping.json`, `sentence_index_mapping.json`, `subsentence_index_mapping.json` |


---

## 4. Graph Label Normalization & Error Standardization

Scripts that standardize labels in graph datasets and normalize error-log formats.

| Script | Main Functions | Goal | Key Outputs |
|--------|----------------|------|-------------|
| **UpdateGraphLabels.py** | Updates graph node labels using a corrected codebook mapping. | Ensure that graph datasets use a unified label system. | Updated graph JSON files. |
| **StandardizeErrorFiles_v1.py** | Standardizes error logs from multiple models into a consistent schema. | Harmonize evaluation outputs across experiments. | `<model>_errors_standardized.json`. |
| **StandardizeErrorFiles_v2.py** | More robust version of the error standardizer that handles missing or variant fields. | Improve consistency and robustness of error-file normalization. | Standardized error JSON files. |
| **Experiments_3_SRL_GCN_DataProcessing.py** | Injects labels + text into SRL graphs; handles filename cleanup | Prepare SRL graph datasets for GCN training | Updated SRL graph JSONs |
| **Experiments_2_Dep_GCN_DataProcessing_new.py** | Injects labels + text for dependency graphs across split types | Standardize graph-label alignment | Updated DEP graph JSONs |
| **Optimize_code_structure_v2_categorize_labels.py** | Uses a ChatOpenAI/LangChain pipeline to classify each label node from the cleaned hierarchy as interactional or goal-oriented with a brief explanation. | Automatically derive coarse intent types for all labels to support hierarchy redesign and downstream modeling. | `EPPC_output_json/classified_intents.json` |
| **Optimize_code_structure.py** | Loads underscore-indexed label names, classifies each as interactional or goal-oriented with ChatOpenAI, and rebuilds an optimized hierarchy with new indices, alias mappings, and stored LLM rationales. | Redesign the label hierarchy into an intent-aware structure with clean indices and alias maps for downstream graph and model use. | `EPPC_output_json/optimized_label_structure.json`, `index_mapping.json`, `llm_classification_log.json`, `alias_map.json` |
| **labels_config.json** | Defines canonical interactional and goal labels plus exact/regex alias lists that map raw annotation label variants onto each canonical intent type. | Provide a reusable configuration for normalizing raw annotation labels into canonical interactional vs goal-oriented classes. | Configuration file consumed by label-mapping and training/evaluation scripts (no direct outputs). |
| **Optimize_code_structure_v2_clean_labels.py** | Cleans underscore-indexed codebook nodes, removes duplicates, reassigns stable indices, and records old→new index mappings. | Produce a clean, deduplicated codebook structure for downstream mapping. | `cleaned_node_names.json`, `index_mapping.json` |
| **Optimize_code_structure_v2_make_dictionary.py** | Splits classified intents into Interactional vs Goal-Oriented across code/subcode/subsubcode levels. | Convert human-corrected intent classifications into per-type dictionaries. | `split_intents_by_type.json` |
| **Optimize_code_structure_v2_split_types.py** | Partitions classified codebook records into Interactional and Goal-Oriented lists with index/label/level. | Generate clean, type-separated label lists for use in annotation mapping. | `split_intents_by_type.json` |
| **Optimize_code_structure_v2_map_annotation_with_code.py** | Loads annotations, mapping table, and intent types; replaces raw labels with codebook labels and attaches type lists. | Normalize annotation label usage and attach label types. | `processed_annotations_with_types.json` |
| **Optimize_code_structure_v2_map_annotation_with_code_v2.py** | Refines annotation mapping: remaps labels, preserves updated codes, attaches label types, and outputs cleaned annotation structures. | Final standardized annotation file linking codes and their Interactional/Goal types. | `processed_messages_with_annotations_with_types.json` |
| **split_annotations_by_label_type.py** | Splits processed annotations into two files (Interactional and Goal-Oriented) and filters codes accordingly. | Produce fully separated annotation datasets per intent type. | `interactional_annotations.json`, `goal_oriented_annotations.json` |


---

## 5. Span Consistency & Sentence/Annotation Structure Evaluation

Scripts focused on span length, alignment to sentence boundaries, and structural quality of annotations.

| Script | Main Functions | Goal | Key Outputs |
|--------|----------------|------|-------------|
| **Bethesda_AnnotationAlignmentStats.py** | Measures alignment between annotations and segmented units for the Bethesda dataset. | Validate clause segmentation quality and alignment. | Summary statistics printed or saved. |
| **Dataset_Stat_4_SpanConsistency.py** | Runs full span inconsistency analysis: span lengths, bucket distributions, sentence-boundary alignment, boundary distances, histogram, and LaTeX snippet. | Quantify span inconsistency and provide evidence for span-level challenges. | `span_stats.csv`, `overall_span_length_stats.json`, `per_label_span_stats.csv`, `span_length_histogram.png`, `span_length_buckets.csv`, `alignment_summary.json`, `boundary_distance_stats.json`, optional LaTeX snippet. |
| **Experiments_2_check_graph_validity.py** | Validates dependency graphs (nodes, labels); logs invalid ones | Ensure structural integrity of DEP graphs | `invalid_graphs_log.json` + filtered folder |
| **Experiments_3_check_graph_validity_srl.py** | Validates SRL graphs (token labels, node lists); normalizes filenames | Filter usable SRL graphs for GCN | Valid SRL graph set + `invalid_graphs_log.json` |

---

## 6. Documentation & Supporting Notes

Markdown files describing pipelines and preprocessing steps.

| File | Description |
|------|-------------|
| **DATA_PREPROCESSING.md** | Documents the full annotation preprocessing pipeline, including codebook parsing, annotation processing, segmentation, evaluation, and visualization steps. |

---


## **7. Decision-Structure Parsing & Phrase-Chunking Tools**
Scripts that generate clause structures, chunk phrases, and support phrase-level scoring or lookup for symbolic narrative/intent modeling.

| Script | Main Functions | Goal | Key Outputs |
|--------|----------------|------|-------------|
| **Decision_Structure_1_chucking.py** | Early version of clause + phrase chunking using spaCy heuristics | Prototype phrase-chunking for structural analysis | Printed phrase lists |
| **Decision_Structure_1_chucking_v1.py** | Revised chunking; improved VP/PP handling | Improve chunk segmentation quality | Printed clause/chunk outputs |
| **Decision_Structure_1_chucking_v2.py** | More robust dependency-based chunk extraction | Stabilize phrase grouping logic | Clause + phrase dict output |
| **Decision_Structure_1_chucking_v3.py** | Final refined chunker: phrasal verbs, copula predicates, PP nesting, discourse markers | Provide high-quality structural chunks for phrase scoring | Clause + phrase JSON/dict outputs |
| **Decision_Structure_1_chucking_checkpoint.py** | Checkpoint version for intermediate debugging | Preserve intermediate chunk-rules for comparison | Console printouts |
| **Decision_Structure_1_chucking_checkpoint_v2.py** | Updated checkpoint with improved span alignment | Debug chunk boundaries against test sentences | Console printouts |
| **Decision_Structure_1_chucking_local_llm.py** | Calls a local LLM to produce phrase/chunk structures; compares with spaCy versions | Test LLM-based structural chunking | Printed LLM vs spaCy comparisons |
| **Decision_Structure_1_look_up.py** | Prototype phrase→label lookup using small index | Test lightweight index retrieval logic | Printed label predictions |
| **Decision_Structure_1_look_up_v2.py** | Improved lookup logic: similarity thresholds, PMI scoring | More stable phrase→label matching | Printed similarity + label scores |

---

## **8. Phrase Lookup & Scoring Engine**
Scripts that build a phrase–label index from CSVs (phrase_pool, label_edges) and score phrases using similarity + PMI.

| Script | Main Functions | Goal | Key Outputs |
|--------|----------------|------|-------------|
| **phrase_lookup_config.py** | Builds phrase–label index; performs fuzzy matching; scores labels using PMI and co-occurrence | Provide interpretable phrase-level label scoring engine | `phrase_index_cache.json`, console scores |
| **phrase_debug_and_lookup.py** | End-to-end: runs Decision_Structure_1_chucking_v3 → prints phrases → performs label lookup → summarizes scores | Debug phrase scoring and inspect per-phrase label contributions | Detailed console diagnostics |
| **Processed_0_DEP_GCN.py** | Loads dependency graph feature dataset, splits into train/val/test, trains a 2-layer GCN for label prediction, logs performance, and saves best model and error/correct cases. | Run dependency-graph GCN classification baseline. | `results_dep_gcn/best_model.pt`, `dep_gcn_errors.json`, `dep_gcn_correct.json`, `dep_gcn_plot.png` |
| **Processed_1_SRL_anchored_gcn.py** | Trains a GCN on anchored SRL graph features with edge weights, filters singleton classes, tracks loss/accuracy/macro-F1, and saves best model and error logs. | Evaluate anchored SRL graph representations with a GCN classifier. | `results_srl_anchored/best_model.pt`, `train_log.json`, `errors.json`, `training_plot.png` |
| **Processed_1_SRL_predicate_gcn.py** | Runs a GCN over predicate-centric SRL graph features, with class filtering, stratified split, and training curve logging. | Test predicate-centric SRL graphs for label prediction using GCN. | `results_srl_predicate/best_model.pt`, `train_log.json`, `errors.json`, `training_plot.png` |
| **Processed_1_SRL_weighted_gcn.py** | Trains a weighted GCN on SRL graphs with edge weights, evaluates with macro-F1, logs metrics, and stores misclassified samples. | Assess edge-weighted SRL graph models via GCN. | `results_srl_weighted/best_model.pt`, `train_log.json`, `errors.json`, `training_plot.png` |
| **Processed_2_AMR_GCN.py** | Trains a GCN on AMR graph features, filters rare classes, performs stratified splitting, and records training curves and errors. | Evaluate AMR-based graph representations with GCN classification. | `results_amr_gcn/best_model.pt`, `train_log.json`, `errors.json`, `training_plot.png` |
| **Processed_5_Narrative_GCN.py** | Loads narrative AMR-ego PyG graphs, trains a GCN, logs loss/acc/F1, evaluates center-node prediction, and saves model + errors. | Narrative ego-graph GCN baseline. | `results_narrative_ego_amr/` (model, logs, errors, plots) |
| **Processed_5_Narrative_GCN_v2.py** | Extended narrative GCN with 120 epochs, dual-axis training plot, and refined evaluation. | Improved GCN variant for narrative graph classification. | `results_narrative_ego_amr/` (model, logs, plots, errors) |
| **Processed_6_narrative_MLP_features.py** | Converts narrative ego-graphs into pooled 768-D BERT features using weighted neighbor aggregation; stores (X, Y, meta). | Generate pooled-feature MSG-MLP dataset. | `processed_graph_features_narrative_ego_mlp.pt` |
| **Processed_6_narrative_MLP.py** | Trains an MLP baseline on pooled narrative features; logs loss/acc/F1, saves model, plot, and error lists. | MSG-MLP baseline classifier. | `results_narrative_ego_mlp/` |
| **Processed_6_narrative_MLP_v2.py** | Improved MSG-MLP with dual-axis plots, validation error analysis, and test-set evaluation. | Enhanced MSG-MLP training + evaluation pipeline. | `results_narrative_ego_mlp_2/` |
| **Processed_6_narrative_MLP_v3.py** | Refined MSG-MLP variant with stable held-out test evaluation and cleaner error tracking. | Third iteration of MSG-MLP focusing on evaluation stability. | `results_narrative_ego_mlp_3/` |
| **Test_Training_0_Linear.py** | Baseline: TF-IDF (word+char) + Logistic Regression using span-or-text input. | Basic linear baseline for sentence/subsentence classification. | Metrics, predictions, confusion matrices under `baseline_step1_outputs/` |
| **Test_Training_0_Linear_v2.py** | Updated version with cleaned input handling and improved evaluation outputs. | Improved TF-IDF baseline pipeline. | Outputs under `baseline_step1_outputs/` |
| **Test_Training_1_Linear.py** | LSA + prototype similarity baseline using span-or-text; stratified split with fallback. | Semantic baseline with dimensionality reduction and prototypes. | Outputs under `baseline_step2_lsa_outputs/` |
| **Test_Training_1_Linear_v2.py** | Improved LSA baseline with full reporting, JSON logs, and cleaner plotting. | Enhanced LSA baseline pipeline. | Outputs under `baseline_step2_lsa_outputs/` |
| **Test_Training_1_Linear_text_only.py** | LSA baseline using **text only** (no span), prototype similarities, improved robustness. | Text-only semantic LSA baseline. | Outputs under `baseline_step2_lsa_text_outputs/` |
| **Training_1_single_Label_GCN.py** | Loads dependency-graph JSONs and structured labels, builds PyG graphs, trains a 3-layer GCN for single-label classification, and evaluates with exact-match and F1 metrics. | Single-label GCN baseline for dependency graphs. | Model weights, logs, plots, and `results/test_predictions.json` |
| **Training_1_Multi_Label_GCN.py** | Loads dependency-graph JSONs and multi-hot structured labels, constructs PyG graphs, trains a 3-layer GCN with class-balanced BCE loss, and evaluates using exact-match, F1, and prediction logs. | Multi-label GCN classifier for dependency graphs. | Model weights, logs, plots, and `results/test_predictions.json` |


---

## **9. Graph-based Modeling**
Scripts that train GCN models on dependency-based clause graphs (tokens as nodes, syntactic edges as edges).

| Script | Main Functions | Goal | Key Outputs |
|--------|----------------|------|-------------|
| **Experiments_2_Dep_GCN.py** | Full DEP-GCN training pipeline with BERT node features, pooling, metrics | Core dependency-graph GCN baseline | Metrics, `best_model.pt`, training plot |
| **Experiments_2_Dep_GCN_v2.py** | Version with fixed random seed, cleaned training logs, and structured outputs | Improve reproducibility and log structure | `train_log.json`, plots, errors |
| **Experiments_2_Dep_GCN_v3.py** | Extended training (500 epochs), updated metadata fields, stable loss curves | Deep training run to test convergence limits | Extended logs, high-resolution plot, structured errors |
| **Experiments_3_SRL_GCN_weighted.py** | SRL-GCN model using weighted edges + BERT CLS vectors for node features | Explore semantic edge weighting in SRL graphs | Metrics, `best_model.pt`, `train_log.json`, plots |
| **Experiments_3_SRL_GCN_weighted_v2.py** | Large-epoch SRL-GCN training with weight-sensitive edges and identity-node encoding | Test stability and long-horizon training for weighted SRL GCN | Metrics, long training trajectory, `best_model.pt`|
| **Experiments_4_AMR_GCN_DataProcessing.py** | Injects subcode labels and texts into AMR graphs; saves updated versions | Prepare labeled AMR graphs for AMR-GCN training | Labeled AMR graph JSON files |
| **Experiments_4_check_graph_validity_amr.py** | Validates node/edge structure of AMR graphs; filters usable ones | Ensure AMR graph correctness before GCN training | `invalid_graphs_log.json`, filtered AMR set |
| **Graph_1_Dependency.py** | Reads sentence/subsentence label files, parses text with spaCy, builds dependency graphs (nodes and links), and saves both JSON graph structures and PNG visualizations. | Construct dependency-based graph representations for labeled sentences/subsentences to support graph modeling and qualitative inspection. | Per-item graph JSON files under `<base_name>/json/` and PNG images under `<base_name>/images/` |
| **Processed_0_DEP_features_labeled.py** | Converts dependency graphs into PyG format using canonical codebook labels, builds node POS features, and saves dataset. | Dependency-graph PyG dataset with canonical labels for classification models. | `dep_graph_features.pt` |
| **Text2Graph_1_construct_graph.py** | Builds sequential narrative graphs grouped by message_id using sentence/subsentence label files; adds "next" edges and saves JSON + visualizations. | Construct ordered narrative graphs from labeled units. | Graph JSONs + PNGs under `outputs/text2graphs_*` |
| **Text2Graph_1_run_all_dependency.py** | Loads text for each sentence/subsentence and builds dependency graphs using DependencyGraphBuilder; saves JSON + PNG. | Full pipeline to generate dependency graphs for all units. | JSON + PNG under `Dep_<input>_graph/` |
| **Text2Graph_1_run_pipeline.py** | Single-example tester for dependency graph construction and visualization. | Quickly test dependency graph building for one example. | On-screen visualization |
| **Text2Graph_2_run_all_srl.py** | Runs AllenNLP SRL over all sentences/subsentences, extracts predicates/arguments, builds semantic graphs, and saves JSON + PNG. | Full SRL-to-graph pipeline for the entire dataset. | Semantic graph JSONs + PNGs under `semantic_graphs/` |
| **Text2Graph_2_run_all_srl_v2.py** | Updated SRL batch processing with refined graph building logic and cleaned output folders. | Improved SRL graph generator with updated format. | Semantic graph JSONs + PNGs |
| **Text2Graph_4_narrative.py** | Builds clause-centered narrative ego-graphs using next/prev/same-label edges; outputs JSON and visualizations. | Construct narrative ego-graphs for each clause. | JSON + PNG under `outputs/narrative_egograph_per_clause/` |
| **Text2Graph_4_narrative_v2.py** | Updated ego-graph builder with corrected edge logic, organized folders, and consistent center-id graph format. | Improved narrative ego-graph generation. | JSON + PNG under `outputs/narrative_egograph_per_clause/` |
| **Text2Graph_4_narrative_v3.py** | Full narrative ego-graph generator with weighted edges, cue-based relations (contrast/elaboration), and detailed visualization. | Final narrative ego-graph pipeline with weighted symbolic relations. | JSON under `json/` and PNG under `images/` |
| **narrative_graph_visualize.py** | Loads narrative graphs and renders Spring-layout PNGs with edge color coding. | Visualize narrative graphs stored in JSON. | PNGs under `outputs/narrative_graphs/.../images/` |
| **Text2Graph_weighted_narrative_graphs.py** | Generates message-level weighted narrative graphs for each clause, using next + same-label + optional cue edges. | Create weighted narrative graphs for each clause instance. | JSON files under `outputs/narrative_weighted_graphs/` |
| **Text2Graph_2_run_all_srl.py** | Runs AllenNLP SRL for each sentence/subsentence; extracts predicate + arguments and saves SRL graphs (JSON + PNG). | Full SRL-to-graph pipeline. | SRL graph JSON + PNG under `outputs/srl_graphs/.../` |
| **Text2Graph_2_run_all_srl_v2.py** | Updated SRL batch processor with improved parsing, graph formatting, and cleaned output directories. | Improved SRL graph builder. | JSON + PNG under `outputs/srl_graphs/.../` |
| **Text2Graph_2_run_all_srl_v3.py** | Enhanced SRL pipeline with richer role descriptions, predicate/argument typing, and color-coded visualization. | Rich SRL graph generation with semantic role metadata. | JSON under `json/`, PNG under `images/` |
| **Text2Graph_2_run_all_srl_v4.py** | A more verbose SRL version with proxy structure and multi-predicate handling; produces predicate-centered graphs. | Advanced SRL graph builder with proxy center node. | JSON + PNG under `outputs/srl_graphs_more_role/.../` |
| **Text2Graph_2_run_all_srl_weight.py** | Zero-shot classification (BART-MNLI) used as a lightweight SRL proxy; assigns weighted edges for semantic roles. | Weighted semantic-proxy graphs for each unit. | Weighted JSON + PNG under `outputs/srl_graphs_weighted/.../` |
| **Experiments_5_Narrative_DataProcessing.py** | Injects labels + text into narrative ego-graphs using center_id mapping | Prepare labeled narrative graphs for GCN modeling | Narrative graph JSON with labels/text |
| **Experiments_5_check_graph_validity_narrative.py** | Validates narrative ego-graph structure (nodes, edges, center_id, label) | Ensure narrative graph integrity | `invalid_graphs_log.json`, filtered narrative graphs|
| **Text2Graph_3_amr.py** | Runs AMR parser on each sentence/subsentence, cleans concept labels, builds AMR graphs, and saves JSON + PNG. | Full AMR graph construction for dataset. | JSON under `outputs/amr_graphs/.../json/`, PNG under `images/` |
| **Text2Graph_3_amr_example.py** | AMR parsing demo for a single sentence with consistent concept remapping and graph visualization. | Example AMR graph parser + visualizer. | JSON + PNG in working folder |
| **Text2Graph_3_amr_example_v2.py** | Multi-sentence AMR example script with cleaned variable handling and visualization. | Batch AMR example generator. | JSON + PNG under `outputs/amr_graph_example/` |
| **Text2Graph_3_amr_model.py** | Loads local AMR parsing model and tokenizer for use in AMR scripts. | AMR model loader for graph construction pipelines. | No standalone outputs |
| **Text2Graph_1_run_pipeline.py** | Builds and visualizes dependency graph for a single example (test script). | Dependency graph demo. | On-screen visualization |


---

## **10. Visualization Tools**
Utilities for generating subtree or structure visualizations.

| Script | Main Functions | Goal | Key Outputs |
|--------|----------------|------|-------------|
| **GenerateSubTreeVisualizations.py** | Loads hierarchy JSON and produces PNGs for each subtree (code/subcode) | Visualize label hierarchy structure for inspection | PNG images under `subtree_images/`|
| **ListNodeNameWIndex.py** | Loads the codebook hierarchy, finds root codes, traverses children with DFS, assigns stable underscore-based indices, and groups names by node type. | Create a stable, human-readable index mapping for all codebook nodes to support lookup, prompting, and visualization. | `EPPC_output_json/node_names_by_type_with_index.json` |
| **message_structure_visualization.py** | Builds hierarchical message→sentence→subsentence graphs, applies layout, and saves PNG visualizations. | Visualize message structural trees for inspection. | PNGs under `structure_visualized_graphs/` |
| **message_structure_visualization_updated.py** | Improved visualization with color coding, updated layouts, and per-message JSON graph exports. | Produce enhanced visual+JSON structural graphs for each message. | PNGs under `visualized_graphs_updated/`, JSONs under `structured_graph_json/` |
| **Text2Graph_1_run_pipeline.py** | Single-example dependency graph tester with direct visualization. | Sandbox visualization of dependency graph. | On-screen plot |
| **Text2Graph_zero_shot_demo.py** | (file with BART large MNLI zero-shot classification → graph) Creates demonstration graph from a sentence using zero-shot labels. | Demo script for lightweight semantic graph building. | `srl_graph_output.png` |


---

## **11. File Conversion Tools**
Scripts for converting Markdown, SPSS .sav, and other formats.

| Script | Main Functions | Goal | Key Outputs |
|--------|----------------|------|-------------|
| **File_transform_md2txt.py** | Converts Markdown files to TXT/DOCX via Pandoc (with fallback parser) | Quickly export documentation into text formats | Converted `.txt` / `.docx` files|
| **File_transform_sav2csv.py** | Converts `.sav` (SPSS) files into UTF-8 CSV via pyreadstat | Extract tabular data from SPSS for analysis | `.csv` files saved to folder|

---

## How to Extend This Document

When adding new scripts, use the following pattern:

```md
| **ScriptName.py** | Key operations | Purpose | Outputs |
