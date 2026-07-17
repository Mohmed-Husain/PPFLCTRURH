# Federated NLP Pipeline Handoff

This directory contains the modular, standardized NLP pipeline built for **Federated Hierarchical Multi-Label Clinical Text Classification**.

## 📁 Directory Structure
```text
federated_handoff/
├── nlp_pipeline/          # Core Python package (modules: configs, data, models, training, api, utils)
├── setup.py               # Package installation configuration
└── README.md              # This integration guide
```

## 🛠️ Installation
The federated learning (FL) simulation coordinator or client nodes can install this module by running:
```bash
pip install -e .
```

---

## 🔌 API Contract for the FL Simulator Team (Ratan & Harshit)

The FL team should interact **exclusively** with the `FederatedNLPClient` class from `nlp_pipeline.api.interface` (or directly imported from `nlp_pipeline`).

### Quick Start Example (Client Node Training Loop)

```python
from nlp_pipeline import PipelineConfig, FederatedNLPClient
from nlp_pipeline.data.hierarchy import CodingSystemHierarchy
from nlp_pipeline.data.label_encoder import HierarchicalLabelEncoder

# 1. Instantiate Configuration (passed from coordinator/client configs)
config = PipelineConfig(
    model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    device="cuda",  # or "cpu"
    batch_size=8,
    learning_rate=2e-5,
    loss_type="combined",  # focal + hierarchical consistency penalty
)

# 2. Set up Shared Label Hierarchy (identical across all clients)
hierarchy = CodingSystemHierarchy.from_json("label_hierarchy.json")
label_encoder = HierarchicalLabelEncoder.load("label_encodings.json", hierarchy)

# 3. Initialize the Federated Client
client = FederatedNLPClient(config, hierarchy, label_encoder)

# ── FL Simulation Round Loop ──
for round_idx in range(num_rounds):
    # Retrieve global weights from FL Server
    global_weights = get_weights_from_coordinator()
    
    # Load global weights into the client
    client.set_weights(global_weights)
    
    # Ingest local private data
    # (data should be preprocessed into standardized dict records: {'text': ..., 'labels': [...]})
    local_records = my_local_database.get_records()
    local_dataloader = client.build_dataloader(local_records, shuffle=True)
    
    # Train locally for 1 epoch
    metrics = client.local_train_epoch(local_dataloader, silent=True)
    print(f"Local training loss: {metrics['train_loss']:.4f}")
    
    # Extract updated weights to send back for aggregation (e.g. FedAvg, FedProx)
    client_weights = client.get_weights()
    num_samples = client.get_num_samples()  # weights size for aggregation
    
    send_to_coordinator(client_weights, num_samples)
```

### Methods Summary

- `set_weights(state_dict: OrderedDict)`: Sets local model parameters. Automatically resets optimizer buffers.
- `local_train_epoch(dataloader: DataLoader) -> dict`: Fits model locally for one epoch.
- `get_weights() -> OrderedDict`: Retrieves updated parameters to send back.
- `get_num_samples() -> int`: Returns training set size.
- `evaluate(dataloader: DataLoader) -> dict`: Runs local testing metrics.
- `predict(texts: list[str]) -> np.ndarray`: Returns predictions probabilities.
