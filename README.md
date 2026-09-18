# **League of Legends Match Outcome Predictor**

**A time-aware machine learning pipeline that predicts the outcomes of professional League of Legends matches.**

---

## Data Source
* **[Oracle's Elixir](https://oracleselixir.com/tools/downloads)**: Compiled and merged **5 annual datasets (2022–2026)**, yielding roughly **597,000 total rows** of data representing **49,000+ unique professional matches**.


## Technologies Used
* **Python (Scikit-learn, XGBoost)**
* **Jupyter Notebook**

  
## Key Features
* Recovered data coverage gaps without biased imputation.
* Engineered rolling team statistics and dynamic Elo ratings.
* Prevented data leakage with chronological train/test splits and `TimeSeriesSplit` cross-validation.
* Precision-recall decision boundaries are optimized.


## Results
* Improved from **0.53 to 0.71** via `GridSearchCV` optimization.
* Decreased false positives by **69%**.


## Usage Guide
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/vphngt/lol-match-outcome-prediction
cd lol-match-predictor
pip install -r requirements.txt
```

