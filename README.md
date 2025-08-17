# 🧬 Breast Cancer Mortality & Survival Prediction using Machine Learning  

This project was developed as part of my **BSc (Hons) Business Information Systems** dissertation at the University of Westminster.  

---

## 📌 Overview  
This project investigates the use of **machine learning** to predict **breast cancer mortality status** and **expected survival time**.  
The aim is to provide healthcare professionals with a **data-driven decision support tool** for treatment planning and patient care.  

- **Dataset:** SEER Breast Cancer Dataset (4,024 patient records)  
- **Algorithms Tested:** Logistic Regression, Linear Regression, Random Forest, XGBoost  
- **Best Model:** XGBoost (Classifier + Regressor)  
- **Performance:**  
  - Mortality classification accuracy → **89%**  
  - Survival time regression MAE → **19.93 months**  

---

## ⚙️ Features  
- Streamlit user interface for easy interaction  
- Predicts both **mortality status** (Alive/Dead) and **survival time**  
- Displays predictions with **clear visualisations**  
- Ensures patient privacy — no PII stored or processed  

---

## 📂 Project Structure

- `data` – Preprocessed dataset
- `models` – Trained ML models (XGBoost classifier & regressor + scaler)
- `Final Year Project.pdf` – Full dissertation report
- `app.py` – Streamlit front-end for predictions
- `.gitignore` – Ignore Gitignore
- `LICENSE` – MIT License
- `requirements.txt` – Requirements to open

---

## 🧑‍⚖️ Disclaimer  
⚠️ This project is for research and educational purposes only. It is not a certified medical tool and should not be used for clinical decision-making.

---

## 🚀 How to Run  
Clone the repository and install dependencies:  

```bash
git clone https://github.com/Bengajadar/breast-cancer-mortality-survival-prediction.git
cd breast-cancer-mortality-survival-prediction
pip install -r requirements.txt
