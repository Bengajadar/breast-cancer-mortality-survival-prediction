# Breast Cancer Mortality & Survival Prediction using Machine Learning

This project was developed as part of my **BSc (Hons) Business Information Systems** dissertation at the University of Westminster.

## 📌 Overview
The project investigates the use of machine learning to predict **breast cancer mortality status** and **expected survival time**.  
The aim is to provide healthcare professionals with a data-driven decision support tool.

- **Dataset:** SEER Breast Cancer Dataset (4,024 patient records)
- **Algorithms Tested:** Logistic Regression, Linear Regression, Random Forest, XGBoost
- **Best Model:** XGBoost (Classifier + Regressor)
- **Performance:**  
  - Mortality classification accuracy: **89%**  
  - Survival time regression MAE: **19.93 months**

## ⚙️ Features
- Streamlit user interface for easy interaction
- Predicts both **mortality status** (Alive/Dead) and **survival time**
- Displays predictions with clear visualisations
- Ensures patient privacy — no PII stored or processed

## 📂 Project Contents
- `app.py` – Streamlit front-end for predictions
- `cleaned_breast_cancer_data.csv` – Preprocessed dataset
- `*.pkl` – Trained ML models (XGBoost classifier & regressor + scaler)
- `Final Year Project.pdf` – Full dissertation report

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
