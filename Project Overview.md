Based on your professor's proposal, your **AAAI-27 paper is an implementation + experimental research paper**, not a review. Your team's objective is to answer one research question:

> **Can privacy-preserving federated learning perform hierarchical clinical text classification under realistic non-IID hospital settings without sacrificing too much accuracy?** 

### High-Level Pipeline

1. **Dataset Preparation**

   * Obtain a clinical text dataset (likely MIMIC-III/IV).
   * Prepare hierarchical ICD labels.
   * Clean and preprocess the text.

2. **Non-IID Data Simulation**

   * Split data into multiple "hospital" clients.
   * Create realistic label and quantity skew across clients.

3. **Clinical NLP Model**

   * Implement a baseline HMLTC (Hierarchical Multi-Label Text Classification) model.
   * This is the core NLP component.

4. **Federated Learning**

   * Train the model using:

     * FedAvg (baseline)
     * FedProx or SCAFFOLD (heterogeneity-aware)

5. **Privacy**

   * Add Differential Privacy.
   * Sweep different privacy budgets (ε values).

6. **Experiments**

   * Compare:

     * Centralized Training
     * Local-only Training
     * Federated Learning
     * Federated + Differential Privacy

7. **Evaluation**

   * Classification metrics (Micro/Macro F1, Precision, Recall, etc.)
   * Hierarchical metrics
   * Communication rounds
   * Privacy vs Utility trade-off

8. **Analysis**

   * Determine:

     * When FL approaches centralized performance.
     * When non-IID data hurts performance.
     * How much accuracy is lost due to privacy.

---

## Suggested Team Split

### Husain + Shreya + Arpit (NLP Team)

* Dataset preparation
* Text preprocessing
* HMLTC implementation
* Model training
* Evaluation metrics
* Result analysis

### Harshit + Ratan (Federated Learning Team)

* Client simulation
* Non-IID partitioning
* FedAvg
* FedProx / SCAFFOLD
* Differential Privacy
* Federated training infrastructure

### Joint Work

* Integrate NLP model with FL framework.
* Run experiments.
* Analyze results.
* Write the AAAI paper.

This split aligns well with the methodology your professor outlined, with your subgroup focusing on the **NLP model** and Harshit/Ratan focusing on the **federated learning and privacy infrastructure**.
