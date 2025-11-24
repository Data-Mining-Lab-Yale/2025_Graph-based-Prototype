    # TYPE_LOOKUP_FILE = "EPPC_output_json/Labels/split_intents_by_type.json"
    # LABEL_FILE_PATH = "EPPC_output_json/sentence_goal_oriented_label.json"
    # GRAPH_DIR = "Dep_sentence_goal_oriented_label/json"

    # LABEL_FILE_PATH = "EPPC_output_json/sentence_interactional_label.json"
    # GRAPH_DIR = "Dep_sentence_interactional_label/json"

    # LABEL_FILE_PATH = "EPPC_output_json/subsentence_goal_oriented_label.json"
    # GRAPH_DIR = "Dep_subsentence_goal_oriented_label/json"

    # LABEL_FILE_PATH = "EPPC_output_json/subsentence_interactional_label.json"
    # GRAPH_DIR = "Dep_subsentence_interactional_label/json"
import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# --- GCN Model Implementation for Multi-Label Classification ---
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        # 3-layer GCN with increased hidden dimension
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        
        self.lin = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers with ReLU and Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = global_mean_pool(x, batch)
        
        x = self.lin(x)
        return x

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(label_file_path, graph_dir, type_lookup_file):
    graphs = []
    # A simple mapping for POS tags to numerical features
    pos_to_int = {
        'INTJ': 0, 'ADV': 1, 'PUNCT': 2, 'X': 3, 'PRON': 4,
        'NUM': 5, 'SPACE': 6, 'VERB': 7, 'ADP': 8, 'NOUN': 9,
        'DET': 10, 'ADJ': 11, 'CCONJ': 12, 'SCONJ': 13,
        'AUX': 14, 'PART': 15, 'PROPN': 16, 'SYM': 17
    }

    try:
        with open(label_file_path, 'r') as f:
            label_data = json.load(f)
        with open(type_lookup_file, 'r') as f:
            type_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Required file not found. Check paths.")
        return [], {}, 0

    label_list = []
    if "goal" in label_file_path.lower():
        label_list = [item["label"] for item in type_data.get("Goal-Oriented", [])]
    elif "interactional" in label_file_path.lower():
        label_list = [item["label"] for item in type_data.get("Interactional", [])]
    else:
        print("Warning: Could not determine label type from file name. Using all labels.")
        label_list = [item["label"] for item in type_data.get("Goal-Oriented", [])] + [item["label"] for item in type_data.get("Interactional", [])]
        
    unique_labels = sorted(list(set(label_list)))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    num_classes = len(label_to_int)

    print(f"Loaded {num_classes} classes for training.")
    print(f"Labels found in lookup file: {unique_labels}")
    print(f"Label map with contiguous indices: {label_to_int}")

    for item_id, item_info in label_data.items():
        graph_filepath = os.path.join(graph_dir, f"{item_id}.json")
        
        if not os.path.exists(graph_filepath):
            print(f"Graph file for ID '{item_id}' not found. Skipping...")
            continue
            
        with open(graph_filepath, 'r') as f:
            graph_json = json.load(f)

        nodes = graph_json.get("nodes", [])
        links = graph_json.get("links", [])
        
        labels_info = item_info.get("labels", [])
        
        if not labels_info:
            continue

        y_multi_hot = torch.zeros(num_classes, dtype=torch.float)
        valid_labels_found = False
        for label_dict in labels_info:
            for label in label_dict.get("label", []):
                if label in label_to_int:
                    y_multi_hot[label_to_int[label]] = 1.0
                    valid_labels_found = True
                else:
                    print(f"Warning: Label '{label}' from '{item_id}' not found in label map. Skipping this label.")
                    
        if not valid_labels_found:
            print(f"Skipping {item_id}: no valid labels found that are in the label map.")
            continue
            
        x_features = []
        for node in nodes:
            pos_int = pos_to_int.get(node['pos'], len(pos_to_int))
            x_features.append([pos_int])
        
        x = torch.tensor(x_features, dtype=torch.float)
        
        edge_index_list = []
        for link in links:
            edge_index_list.append([link['source'], link['target']])
        
        if not edge_index_list:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        graphs.append(Data(x=x, edge_index=edge_index, y=y_multi_hot.unsqueeze(0)))

    return graphs, label_to_int, num_classes

# --- Training, Evaluation, and Testing ---
def create_data_loaders(graphs, train_ratio=0.7, eval_ratio=0.15):
    if not graphs:
        raise ValueError("No graphs were loaded. Please check your data directory.")
    
    train_indices_temp, test_indices = train_test_split(
        np.arange(len(graphs)), train_size=train_ratio + eval_ratio
    )
    
    train_graphs_temp = [graphs[i] for i in train_indices_temp]
    test_graphs = [graphs[i] for i in test_indices]
    
    train_indices, eval_indices = train_test_split(
        np.arange(len(train_graphs_temp)), train_size=train_ratio / (train_ratio + eval_ratio)
    )
    
    train_graphs = [train_graphs_temp[i] for i in train_indices]
    eval_graphs = [train_graphs_temp[i] for i in eval_indices]

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    return train_loader, eval_loader, test_loader

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    predictions, ground_truths = [], []
    
    num_graphs = len(train_loader.dataset)
    pos_counts = torch.sum(torch.cat([g.y for g in train_loader.dataset]), dim=0)
    neg_counts = num_graphs - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-5)
    pos_weight = pos_weight.to(device)
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        
        loss = F.binary_cross_entropy_with_logits(out, data.y, pos_weight=pos_weight)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        preds_binary = (torch.sigmoid(out) > 0.5).int()
        predictions.extend(preds_binary.cpu().numpy())
        ground_truths.extend(data.y.cpu().numpy())
    
    f1_micro = f1_score(ground_truths, predictions, average='micro', zero_division=0)
    exact_match_acc = np.mean(np.all(np.array(predictions) == np.array(ground_truths), axis=1))

    return total_loss / len(train_loader), exact_match_acc, f1_micro

def evaluate_model(model, loader, device):
    model.eval()
    predictions, ground_truths = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds_binary = (torch.sigmoid(out) > 0.5).int()
            predictions.extend(preds_binary.cpu().numpy())
            ground_truths.extend(data.y.cpu().numpy())
            
    f1_micro = f1_score(ground_truths, predictions, average='micro', zero_division=0)
    exact_match_acc = np.mean(np.all(np.array(predictions) == np.array(ground_truths), axis=1))
    
    return exact_match_acc, f1_micro

def test_model(model, loader, int_to_label, device):
    model.eval()
    predictions_log = []
    predictions, ground_truths = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds_binary = (torch.sigmoid(out) > 0.5).int()
            predictions.extend(preds_binary.cpu().numpy())
            ground_truths.extend(data.y.cpu().numpy())
            
            for i in range(len(data.y)):
                true_labels = [int_to_label[idx] for idx, val in enumerate(data.y[i]) if val == 1]
                pred_labels = [int_to_label[idx] for idx, val in enumerate(preds_binary[i]) if val == 1]
                predictions_log.append({
                    "ground_truth": true_labels,
                    "prediction": pred_labels,
                    "is_correct": sorted(true_labels) == sorted(pred_labels)
                })

    f1_micro = f1_score(ground_truths, predictions, average='micro', zero_division=0)
    exact_match_acc = np.mean(np.all(np.array(predictions) == np.array(ground_truths), axis=1))
    
    return exact_match_acc, f1_micro, predictions_log

if __name__ == "__main__":
    # --- Configuration ---
    # Change these paths and parameters for your specific data and task
    # You can uncomment the desired paths for the data you want to use.

    TYPE_LOOKUP_FILE = "EPPC_output_json/Labels/split_intents_by_type.json"
    
    # Example 1: Sentence-level, Goal-Oriented
    # LABEL_FILE_PATH = "EPPC_output_json/sentence_goal_oriented_label.json"
    # GRAPH_DIR = "Dep_sentence_goal_oriented_label/json"
    
    # Example 2: Sentence-level, Interactional
    LABEL_FILE_PATH = "EPPC_output_json/sentence_interactional_label.json"
    GRAPH_DIR = "Dep_sentence_interactional_label/json"
    
    # Example 3: Subsentence-level, Goal-Oriented
    # LABEL_FILE_PATH = "EPPC_output_json/subsentence_goal_oriented_label.json"
    # GRAPH_DIR = "Dep_subsentence_goal_oriented_label/json"
    
    # Example 4: Subsentence-level, Interactional
    # LABEL_FILE_PATH = "EPPC_output_json/subsentence_interactional_label.json"
    # GRAPH_DIR = "Dep_subsentence_interactional_label/json"

    EPOCHS = 200
    LEARNING_RATE = 0.01
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Main Execution ---
    all_graphs, label_to_int, NUM_CLASSES = load_and_preprocess_data(LABEL_FILE_PATH, GRAPH_DIR, TYPE_LOOKUP_FILE)
    if not all_graphs:
        print("No graphs were loaded. Exiting.")
        exit()
        
    INT_TO_LABEL = {v: k for k, v in label_to_int.items()}
    
    train_loader, eval_loader, test_loader = create_data_loaders(all_graphs)

    num_node_features = all_graphs[0].x.shape[1]
    model = GCN(num_node_features=num_node_features, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_exact_accuracies = []
    eval_exact_accuracies = []
    
    print("Starting GCN training...")
    for epoch in range(EPOCHS):
        train_loss, train_exact_acc, train_f1 = train_model(model, train_loader, optimizer, device)
        eval_exact_acc, eval_f1 = evaluate_model(model, eval_loader, device)
        
        train_losses.append(train_loss)
        train_exact_accuracies.append(train_exact_acc)
        eval_exact_accuracies.append(eval_exact_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Train Exact Acc: {train_exact_acc:.4f}, Eval Exact Acc: {eval_exact_acc:.4f}, Train F1: {train_f1:.4f}, Eval F1: {eval_f1:.4f}')

    # --- Plotting Results ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_exact_accuracies, label='Training Exact Accuracy')
    plt.plot(eval_exact_accuracies, label='Evaluation Exact Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Exact Accuracy')
    plt.title('Exact Accuracy over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    print("Saved training metrics plot as 'training_metrics.png'.")

    # --- Final Test and Prediction Reporting ---
    test_exact_acc, test_f1, predictions_log = test_model(model, test_loader, INT_TO_LABEL, device)
    print(f'\nFinal Test Exact Accuracy: {test_exact_acc:.4f}')
    print(f'Final Test F1-Score (Micro): {test_f1:.4f}')

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "test_predictions.json", 'w') as f:
        json.dump(predictions_log, f, indent=2)
    print("Saved test predictions to 'results/test_predictions.json'.")