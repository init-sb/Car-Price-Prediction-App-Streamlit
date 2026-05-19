# 🚗 Car Price Predictor

A machine learning web app that estimates used car prices based on specifications such as brand, year, mileage, engine size, fuel type, and transmission.

**Live Demo → [car-price-prediction-dsb.streamlit.app](https://car-price-prediction-dsb.streamlit.app)**

---

## Features

- **Predict Price** — instant price estimate with confidence range
- **Model Evaluation** — R² score, MAE, RMSE, MAPE, and feature importance
- **EDA** — scatter plot, histogram, and correlation heatmap with insights
- **Description** — full breakdown of the ML pipeline and tech stack

---

## Model

| | |
|---|---|
| Algorithm | Random Forest Regression |
| R² Score | 0.804 |
| MAE | $1,868 |
| MAPE | 7.98% |

---

## Run Locally

### Prerequisites
- Python 3.9 or higher
- Git

### Windows

```bash
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Mac / Linux

```bash
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open your browser and go to `http://localhost:8501`

---

## Tech Stack

`Python` `Scikit-learn` `Streamlit` `Pandas` `Matplotlib` `Seaborn`