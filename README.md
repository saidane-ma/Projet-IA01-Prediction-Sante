# 🧠 AI01 Project: Health Problem Prediction

## 📋 Project Overview
This project, carried out for the AI01 course (ENSTA Paris / ENIT), aims to develop a **supervised machine learning model** capable of predicting an individual's **health problem level** (`None`, `Low`, `Moderate`) based on various health and lifestyle features.

The entire pipeline—from **data preprocessing** to **final prediction generation**—was implemented entirely in **Python**, using **only the custom modules provided in the AI01 course** (`ia01/utils.py`, `ia01/arbre.py`, etc.), without relying on external libraries like scikit-learn.

**[See the full project report (Rapport_Projet_IA.pdf) for a detailed analysis (in French).]**

---

## ⚙️ Key Features
* **Data Preprocessing:**
    * Missing value imputation (e.g., `age` replaced with train median, `sexe` with alternating imputation).
    * Numerical encoding (Label Encoding) for ordinal (e.g., `sommeil_qualite`) and binary (e.g., `cigarette`) attributes.
    * Feature selection and removal of irrelevant columns (e.g., `pays`, `profession`).

* **Class Imbalance Handling:**
    * The rare `Severe` class (<1% of samples) was merged with `Moderate` to create a more stable 3-class classification problem.

* **Model Selection:**
    * Comparison between **Decision Tree** and **k-Nearest Neighbors (k-NN)** models.
    * **5-fold Cross-Validation** for robust model evaluation.
    * **Macro F1-Score** used as the primary performance metric to ensure a balanced evaluation across all classes (including minorities).

## 🏆 Results
* **Final Model Chosen:** Decision Tree.
* **Optimal Depth:** 3 (identified by cross-validation as the best balance of performance and simplicity).
* **Performance:** This model achieved the best **Macro F1-Score**, outperforming the k-NN model which showed a higher error rate.

---

## 🧩 Repository Structure
## 🧩 Repository Structure

Here is the project structure, designed for modularity and readability:

```text
📂 Project-IA01-Health/
│
├── data/                 # Raw data
│   ├── X_train.csv
│   ├── y_train.csv
│   └── X_test.csv
│
├── ia01/                 # Modules (Tools)
│   ├── __init__.py
│   ├── utils.py
│   ├── arbre.py
│   └── ... (metriques.py, evaluation.py, etc.)
│
├── src/                  # Project source code
│   ├── preprocess.py         # Loading and cleaning functions
│   ├── model_selection.py    # Cross-validation function
│   └── train_predict.py      # Main pipeline script
│
├── results/              # Folder for predictions
│   └── y_test.csv            # Final submission file
│
├── Rapport_Projet_IA.pdf # The analysis report (French)
└── README.md             # This file
 ```

---

## 🧪 Technologies Used
* **Language:** Python 3
* **Libraries:** Custom IA01 modules only (`ia01.utils`, `ia01.arbre`, `ia01.metriques`...)
* **Models:** Decision Tree, k-Nearest Neighbors (k-NN)
* **Metrics:** Macro F1-Score, Error Rate

---

## 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone https://github.com/saidane-ma/Projet-IA01-Prediction-Sante.git
    cd Projet-IA01-Prediction-Sante
    ```

2.  Run the complete pipeline (training and prediction):
    ```bash
    python src/train_predict.py
    ```

3.  The final predictions will be automatically generated and saved in:
    ```
    results/y_test.csv
    ```
