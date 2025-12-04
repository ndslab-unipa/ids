# Multi-layer Intrusion Detection System (IDS)

This project implements a multi-layer Intrusion Detection System based on a **Stacked Ensemble** architecture. The primary goal is to accurately classify network traffic, separating benign flows from various types of attacks.

## IDS Architecture Overview

The system operates in a sequential, four-layer process to maximize detection accuracy and leverage the strengths of different machine learning models. 

### 1. Gatekeeper (Binary Filter)
This module acts as the first line of defense, performing a rapid binary classification. Its goal is to filter out a majority of `BENIGN` traffic, ensuring that the more complex, multi-class experts only process samples likely to be attacks, so this reduces computational overhead. After the first filtering step, only samples classified as **Attack** (binary label 1) are passed to the next stage.

### 2. Expert Classifiers (Multi-Class Detection)
This module is composed of specialized models trained to distinguish between **all possible attack types** (multi-class classification). Each expert outputs predicted **probabilities** and **log-probabilities** for all attack classes.

### 3. Meta-Classifier and Meta-Dataset Generation
This layer utilizes **Stacked Generalization** to learn how to optimally combine and correct the errors of the Experts. Firstly, training data for the Meta-Classifier is not the raw network features, but features derived from the Experts' outputs, including:
* Raw probability distributions of each Expert.
* **Weighted Probabilities:** Probabilities scaled by the Expert's pre-computed, class-specific F1-scores.
* **Entropies:** A measure of uncertainty for each Expert's probability distribution.

### 4. Final Weighted Voting (Heuristic-based Confidence)
The final decision integrates the Ensemble (Experts' aggregated output) and the Meta-Classifier's prediction using a confidence-based heuristic. In **Heuristic Score ($S$)**  the system selects the prediction (Ensemble or Meta) with the higher score, calculated as:

$$S = (\gamma \times \text{F1-Score}_\text{Class}) + ((1 - \gamma) \times \text{Confidence})$$

Where:
* $\text{F1-Score}_\text{Class}$: Pre-calculated F1 performance of the model (Ensemble or Meta) for the predicted class.
* $\text{Confidence}$: Calculated as $1 - \text{Entropy}$.
* $\gamma$: A configurable weighting factor (default 0.5) balancing performance vs. certainty.

This score is computed for both the Ensemble and the Meta-classifier, in order to compute which heuristic score is higher and so output the highly probabilistic prediction.

---

## IDS Project Organization

This repository contains all components necessary for this project inside the `ids/` folder.
| Component | Files/Directory | Purpose |
| :--- | :--- | :--- |
| **Core IDS Logic** | `ids.py` | Main class (`MultiLayerClassifierIDS`) implementing the before mentioned core modules. |
| **Utility Functions** | `utils_ids.py` | Functions for data preprocessing (`binarize`), metric computation, Entropy calculation, and Weighted Probability computation. |
| **Trained Models** | `models/` | Stores all persistent models (`.joblib`, `.h5`) and the computed mean weights (`expert_dt_weights.csv`, `meta_weights.csv`, etc.). |
| **Datasets** | `data/` | Directory for the input datasets. |

---

## Requirements
This project requires Python3.x and the following dependencies.

### Prerequisites
Ensure you have pip (Python package installer) installed.

### Installation
All necessary dependencies can be installed using the `requirements.txt` file. These dependecies can be installed by running the following command in the terminal:
```bash
pip install -r requirements.txt
```
