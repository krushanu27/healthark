# AI-Powered Readmission Risk Prediction

## 🏆 Hackathon Project for RWEsearch & Health Innovation Summit 2025

This project is a submission for the RWEsearch Hackathon by Healthark. It leverages Centers for Medicare & Medicaid Services (CMS) data to build an AI-driven solution that predicts 30-day hospital readmission risk for beneficiaries. [cite_start]The final output is an interactive Streamlit dashboard designed to provide actionable insights for healthcare providers. [cite: 27]



---

## 🚀 The Solution
[cite_start]The core of this project is a machine learning model that analyzes patient demographics, treatment histories, and prescription data to identify individuals at high risk of readmission. [cite: 26] The final model is a fine-tuned **XGBoost Classifier** that achieved an outstanding **97% recall** and a **63% F1-score**, demonstrating a powerful ability to find at-risk patients.

### Key Features:
* **Data Integration**: Combines beneficiary summaries, inpatient claims, outpatient claims, and prescription drug events into a single, analysis-ready dataset.
* **Advanced Feature Engineering**: Creates over 40 features, including patient-level aggregations, one-hot encoded categoricals, and sophisticated provider-level performance metrics.
* **Robust Model Training**: Employs a rigorous pipeline including hyperparameter tuning (`GridSearchCV`) and advanced techniques to handle class imbalance (`scale_pos_weight`).
* **Interactive Dashboard**: A user-friendly Streamlit application that visualizes the model's key insights and performance metrics.

---

## 🛠️ How to Run the Dashboard

**1. Clone the repository:**
```bash
git clone [<your-repo-url>](https://github.com/krushanu27/healthark.git)
cd healthark
```

**2. Install the required libraries:**
```bash
pip install -r requirements.txt
```

**3. Place the data files:**
This project uses the CMS Linkable 2008–2010 Medicare Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF). Due to their size, the data files are not included in this repository. Please download them and place them in the project's root directory.

**4. Run the Streamlit app:**
```bash
streamlit run dashboard.py
```
---

## 📈 Model Performance
After extensive experimentation with multiple architectures (Random Forest, Neural Networks) and data balancing techniques (SMOTE, class weighting), the champion model was a tuned XGBoost Classifier with the following performance on the test set:

| Metric | Score |
| :--- | :--- |
| **F1-Score (Readmitted)** | **63%** |
| **Recall (Readmitted)** | **97%** |
| **Precision (Readmitted)** | 47% |
| **Overall Accuracy** | 94% |

The model's ability to identify 97 out of 100 at-risk patients makes it a highly valuable tool for clinical intervention.
