# Company Bankruptcy Prediction & Clustering (CS559)

A hybrid machine learning project that predicts company bankruptcy based on financial ratios and indicators. The project implements a unique pipeline combining unsupervised clustering (K-Means) to group companies into distinct profiles, followed by training subgroup-specific ensemble stacking models to perform binary classification.

---

## 🚀 Methodology & Pipeline Overview

The project follows a structured data science pipeline divided into three main phases:

### 1. Global Feature Engineering & Selection
* **Data Cleansing**: Columns with zero variance or constant values (e.g., `Net_Income_Flag`) are removed.
* **Correlation Filtering**: Highly correlated features (Pearson correlation coefficient > `0.90`) are filtered out to reduce multicollinearity and improve model generalization.
* **ANOVA F-value Selection**: `SelectKBest` with the `f_classif` scoring function is applied to identify the top 50 most statistically significant features relative to the target `Bankrupt?` variable.
* **Standardization**: Features are normalized using `StandardScaler` to ensure optimal clustering and model training.

### 2. Company Clustering (K-Means)
To identify common risk patterns across different financial structures, the training data is partitioned into **9 distinct subgroups** using K-Means clustering (K=9, `random_state=42`). 

#### Training Data Subgroup Statistics:
* **Subgroup 1** was identified as containing **0 bankrupt companies** (100% non-bankrupt) in the training dataset. Therefore, it is designated as a constant-prediction cluster (always predicts 0).
* The remaining subgroups contain varying proportions of bankrupt/non-bankrupt companies and are modeled independently.

### 3. Subgroup-Specific Ensembles (Stacking Classifiers)
For each cluster (except Subgroup 1), customized stacking classifiers are trained on cluster-specific features and scaling matrices. This allows the models to learn decision boundaries that are highly tailored to the specific financial characteristics of that subgroup.

---

## 📁 Repository Structure

```bash
├── 3_Generalization.ipynb        # Generalization test-set prediction pipeline
├── 3_TrainingData.ipynb          # Feature engineering, clustering, and training pipeline
├── 3_Generalization.csv          # Generated bankruptcy predictions on test data
│
# Subgroup-Specific Jupyter Notebooks
├── Subgroup0.ipynb
├── Subgroup2.ipynb
├── Subgroup3.ipynb
├── Subgroup4.ipynb
├── Subgroup5.ipynb
├── Subgroup6.ipynb
├── Subgroup7.ipynb
├── Subgroup8.ipynb
│
└── .gitignore                     # Git ignore file (excludes heavy video files/caches)
```

*(Note: Stored models (`.pkl` files) and local datasets (`train_data.csv`, `test_data.csv`) are loaded locally and are not tracked in remote storage due to size constraints).*

---

## 🛠️ Setup & Installation

### Prerequisites
Make sure you have Python 3.8+ installed along with Jupyter Notebook or JupyterLab.

### Install Dependencies
Install the required packages using pip:
```bash
pip install pandas numpy scikit-learn joblib jupyter
```

---

## 💻 How to Run

1. **Model Training**: 
   Open and run the cells in `3_TrainingData.ipynb` to execute the feature selection, generate the K-Means cluster labels, and export the clustering files.
   
2. **Subgroup Optimization**:
   Examine the individual subgroup notebooks (e.g., `Subgroup4.ipynb`) to see the custom models, hyperparameter tuning, and stacking classifier implementations for each subgroup.

3. **Generalization (Inference)**:
   Run the `3_Generalization.ipynb` notebook to predict the bankruptcy status of companies in the test dataset. The notebook:
   * Loads the test set.
   * Predicts the cluster label for each company.
   * Dispatches the company to its corresponding subgroup model.
   * Outputs the final prediction mapping to `3_Generalization.csv`.
