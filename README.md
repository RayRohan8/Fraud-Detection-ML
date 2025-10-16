# 🕵️‍♂️ Fraud Detection in Financial Transactions (Machine Learning Project)

Detecting fraudulent credit card transactions using supervised machine learning on highly imbalanced financial data.  
This project builds, evaluates, and compares **Logistic Regression**, **Random Forest**, and **LightGBM** models to identify fraud based on subtle, multidimensional patterns.

---

## 📘 Overview

Financial fraud detection is a classic example of an **imbalanced classification problem**, where fraudulent transactions make up less than 0.2% of all records.  
Our goal is to develop a machine learning model that can:
- Accurately **detect fraudulent transactions** (high recall),
- While minimizing **false alarms** (high precision).

The dataset contains anonymized PCA-transformed features (`V1–V28`), transaction `Time`, and `Amount`.  
We evaluate multiple models and tune thresholds to find the best balance between **catching fraud** and **avoiding false positives**.

---

## 📊 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Samples:** 284,807 transactions
- **Fraud cases:** 492 (≈ 0.17%)
- **Features:**
  - `Time` – seconds elapsed since the first transaction.
  - `Amount` – transaction amount (EUR).
  - `V1–V28` – PCA-transformed anonymized features.
  - `Class` – target (0 = Normal, 1 = Fraud).

⚠️ The dataset is **not uploaded** here due to size and licensing.  
Download from Kaggle and place it in a `data/` folder or Google Drive before running.

---

## ⚙️ Methods

### 🔹 Exploratory Data Analysis (EDA)
- Checked class imbalance (≈ 0.17% fraud).  
- Compared distributions of `Amount`, `Time`, and PCA features (`V1–V28`).  
- Used **Kolmogorov–Smirnov tests** to find most discriminative features.  
- Visualized **PCA 2D projections** to reveal partial separation of fraud vs normal.

### 🔹 Modeling
| Model | Key Traits | Notes |
|-------|-------------|-------|
| **Logistic Regression** | Linear baseline | Interpretable, but limited for complex patterns. |
| **Random Forest** | Ensemble of decision trees | Captures nonlinear relations, good baseline for imbalance. |
| **LightGBM** | Gradient boosting trees | High performance, best suited for structured fraud data. |

- Applied **class weights** to handle imbalance.  
- Evaluated using **ROC-AUC** and **PR-AUC** (Precision-Recall).  
- Tuned decision **thresholds (0.05–0.5)** to study recall–precision trade-offs.  

---

## 📈 Results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | Notes |
|-------|:-------:|:------:|:---------:|:------:|:--------:|-------|
| Logistic Regression | 0.972 | 0.719 | 0.77 | 0.84 | 0.80 | Baseline linear model |
| Random Forest | 0.947 | 0.858 | 0.86 | 0.82 | 0.84 | Strong tree-based model |
| **LightGBM** | **0.977** | **0.889** | **0.88** | **0.85** | **0.86** | ✅ Best overall |

- At **threshold = 0.1**, LightGBM detects ~85% of frauds with ~88% precision.  
- **PR-AUC** is emphasized (fraud = minority class).  
- Confusion matrix confirms low false-positive rate and high fraud detection coverage.

---

## 🧠 Interpretation

- Fraud and normal transactions show **overlapping but shifted distributions** in several PCA components.  
- Tree-based models can exploit **nonlinear combinations** of weak signals (like V12 + Amount).  
- **LightGBM** effectively learns from these subtle patterns, outperforming simpler models.

---

## 🚀 How to Run

1. Clone the repository
   
   git clone https://github.com/<your-username>/Fraud-Detection-ML.git
   cd Fraud-Detection-ML

2. Install dependencies

   pip install -r requirements.txt
   
3. Download the dataset

   From Kaggle
   Place creditcard.csv inside a folder named data/ or mount Google Drive in Colab.
   
4. Run the notebook

   jupyter notebook notebooks/Credit_Card_Fraud.ipynb



