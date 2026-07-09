# AAAI-27 Research Workflow
## Privacy-Preserving Federated Learning for Clinical Text under Realistic Heterogeneity

## Objective

The objective is to maximize parallel development while minimizing idle time between the **NLP Team** and the **Federated Learning (FL) Team**.

---

# Team Responsibilities

## NLP Team
**Members:** Husain, Shreya, Arpit

Responsible for:
- Dataset preparation
- Clinical NLP pipeline
- HMLTC implementation
- Model training
- Evaluation pipeline
- Documentation

---

## Federated Learning Team
**Members:** Harshit, Ratan

Responsible for:
- FL infrastructure
- Client simulation
- Non-IID partitioning
- Differential Privacy
- Federated optimization
- Communication framework

---

# Phase 1 — Independent Setup (Parallel)

> **Dependency:** None

Both teams can start immediately.

---

## NLP Team

### Goal
Prepare a production-ready NLP pipeline that can later plug into any FL framework.

### Tasks

- Study the assigned paper(s) and identify the baseline HMLTC architecture.
- Finalize dataset selection (MIMIC-III / MIMIC-IV or professor-approved dataset).
- Download dataset.
- Understand licensing/access requirements.
- Build preprocessing pipeline:
  - Text cleaning
  - Tokenization
  - Label preprocessing
  - Hierarchical ICD mapping
- Create train/validation/test split.
- Define evaluation metrics:
  - Micro F1
  - Macro F1
  - Precision
  - Recall
  - Hierarchical metrics
- Design project folder structure.
- Prepare configuration files.

### Deliverables

- Clean dataset
- Preprocessing pipeline
- Dataset documentation
- Ready-to-train dataset

---

## Federated Learning Team

### Goal
Prepare the complete federated learning framework independent of the NLP model.

### Tasks

- Study FedAvg
- Study FedProx
- Study SCAFFOLD
- Select FL framework (Flower / FedML / Custom)
- Design overall FL architecture.
- Implement client-server communication.
- Design configuration system.
- Prepare logging system.
- Implement checkpointing.
- Design experiment pipeline.

### Deliverables

- Working FL framework
- Client-server communication
- Experiment configuration
- Logging utilities

---

# Phase 2 — Parallel Development

> **Dependency:** Phase 1 Completed

Both teams continue independently.

---

## NLP Team

### Goal
Develop and validate the standalone NLP model.

### Tasks

- Implement baseline HMLTC model.
- Train centrally.
- Hyperparameter tuning.
- Validate training.
- Save checkpoints.
- Export model weights.
- Export tokenizer.
- Export inference pipeline.
- Build evaluation scripts.
- Document model inputs/outputs.

### Deliverables

- Trained HMLTC model
- Saved checkpoints
- Evaluation scripts
- Inference API

---

## Federated Learning Team

### Goal
Develop federated learning independent of the NLP model.

### Tasks

- Implement client creation.
- Implement non-IID partitioning:
  - Label skew
  - Quantity skew
- Implement FedAvg.
- Implement FedProx.
- Implement SCAFFOLD.
- Build communication round logic.
- Implement client aggregation.
- Test FL pipeline using a dummy model.
- Verify convergence.

### Deliverables

- Fully working FL pipeline
- Non-IID simulator
- Aggregation module
- Tested FL framework

---

# Phase 3 — Integration

> **Dependency:** Phase 2 Completed

This is the first phase where both teams interact.

---

### NLP Team provides

- Final trained model architecture
- Model weights
- Tokenizer
- Training configuration
- Input/output interface

---

### FL Team integrates

- Replace dummy model with HMLTC model.
- Verify compatibility.
- Fix interface issues.
- Execute initial federated training.
- Validate synchronization.

### Deliverables

- Federated HMLTC training working successfully

---

# Phase 4 — Privacy Module

> **Dependency:** Federated training operational

## Federated Learning Team

### Tasks

- Implement Differential Privacy.
- Configure privacy budgets (ε).
- Add privacy accounting.
- Verify training stability.

### Deliverables

- DP-enabled FL pipeline

---

## NLP Team (Parallel)

While DP is being integrated:

### Tasks

- Prepare experiment scripts.
- Prepare evaluation scripts.
- Prepare visualization scripts.
- Prepare result tables.
- Prepare benchmark templates.

### Deliverables

- Automated evaluation pipeline
- Result generation scripts

---

# Phase 5 — Experimental Evaluation

> **Dependency:** DP Integration Complete

Both teams collaborate.

### Experiments

Run:

- Centralized Training
- Local-only Training
- FedAvg
- FedProx
- SCAFFOLD
- FL + Differential Privacy

Vary:

- Number of clients
- Non-IID severity
- Privacy budget (ε)
- Communication rounds

Collect:

- Accuracy
- Precision
- Recall
- Micro F1
- Macro F1
- Hierarchical metrics
- Communication cost
- Training time

---

# Phase 6 — Analysis

Joint responsibility.

Tasks

- Compare centralized vs federated.
- Analyze privacy vs utility.
- Analyze effect of non-IID data.
- Compare FedAvg/FedProx/SCAFFOLD.
- Generate plots.
- Build tables.
- Identify research insights.

Deliverables

- Final experimental results
- Figures
- Tables
- Research conclusions

---

# Phase 7 — AAAI Paper Writing

Joint responsibility.

Suggested ownership:

## NLP Team

- Abstract
- Introduction
- Related Work
- Dataset
- NLP Methodology
- Evaluation Metrics

---

## Federated Learning Team

- Federated Learning Methodology
- Differential Privacy
- Experimental Setup
- Communication Protocol

---

## Joint

- Results
- Discussion
- Limitations
- Future Work
- Conclusion
- Final proofreading

---

# Parallel Workflow Summary

| Phase | NLP Team | FL Team | Dependency |
|--------|----------|---------|------------|
| Phase 1 | Dataset, preprocessing, metrics | FL framework, architecture | None (Parallel) |
| Phase 2 | HMLTC implementation & training | FedAvg, FedProx, SCAFFOLD, Non-IID | Parallel |
| Phase 3 | Deliver trained model | Integrate HMLTC into FL | NLP → FL |
| Phase 4 | Evaluation pipeline & visualization | Differential Privacy integration | Parallel |
| Phase 5 | Run experiments & evaluate | Run federated experiments | Joint |
| Phase 6 | Analyze results | Analyze FL behavior | Joint |
| Phase 7 | Write NLP sections | Write FL sections | Joint |

---

# Critical Handoff Points

### NLP → FL

- Preprocessed dataset
- Label mapping
- Model architecture
- Trained weights
- Tokenizer
- Inference interface

---

### FL → NLP

- Federated checkpoints
- Privacy-enabled models
- Client-wise metrics
- Communication statistics

---

# Expected Outcome

Following this workflow ensures:

- ✅ Maximum parallel development
- ✅ Minimal team blocking
- ✅ Clear ownership
- ✅ Faster integration
- ✅ Efficient experimentation
- ✅ Reduced overall development time