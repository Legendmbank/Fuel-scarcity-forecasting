# Fuel-scarcity-forecasting (MVP) — 3MTT Capstone Project
MVP model for predicting fuel scarcity likelihood using time-series machine learning.

## 1. Project Context & Problem Statement
Fuel scarcity in dynamic markets creates severe distribution bottlenecks and economic disruption. Traditional management relies on reactive responses after shortages occur. This MVP model predicts the **likelihood of fuel scarcity 1 week in advance** to enable proactive inventory and supply planning.

### Relevance to the Nigerian Fuel Market
Although trained across multi-regional market data to capture diverse supply-shock scenarios, the modeling framework is explicitly tailored to Nigeria's macro-environment:
* **Grouped Time-Series Alignment:** All lag features and target shifts were computed `groupby('Country')` to ensure each country's chronological sequence remains isolated from other regional data.
* **Import & FX Vulnerability:** High reliance on import landing costs is captured through `Exchange_Rate_vs_USD`, `Crude_lag_1`, and `Inflation_Rate (%)`.
* **Policy Regimes:** Captures the structural impact of state interventions via `Subsidy_Level_Encoded`.
* **Proactive Forecasting:** Predicts scarcity 1 week in advance using historical lags, providing actionable lead-time for Nigerian downstream regulators and supply chain managers.
  
---

## 2. Data Preparation & Exploratory Data Analysis (EDA)
The raw dataset (`fuel_raw_data.csv`) underwent rigorous cleaning and exploratory analysis in the Data Preparation pipeline:

* **Data Cleaning & Missing Value Handling:**
  * Categorical missing values (e.g., `Event_Description`) were filled with `'No Event'`.
  * Groupwise forward-fill and back-fill (grouped by `Country`) were applied to handle missing values in time-series features (`Petrol_28d_vol`, `Petrol_lag_1`, `Crude_lag_1`).
* **Scarcity Target Engineering:**
  * Defined a critical scarcity condition where **Demand Index > Supply Index** AND **Supply Index < 40**.
  * A forward shift (`shift(-1)`) was applied groupwise to form `Scarcity_Target` (predicting if scarcity will occur in the following week).
* **Time-Based Feature Extraction:**
  * Datetime attributes were converted to derive temporal components: `Year`, `Month`, `Quarter`, `Week`, and `Day_Of_Week`.
* **Exploratory Visual Insights:**
  * **Seasonal Trends:** Analyzed monthly and quarterly distribution patterns, identifying seasonal demand surges against supply dips.
  * **Correlation Analysis:** Evaluated heatmaps across fuel prices, macroeconomic indicators (Exchange Rates, Inflation, Crude Prices), and supply/demand metrics to isolate predictive signals.
  * **Processed Export:** Cleaned and engineered features were saved to `fuel_processed_data.csv`.

---

## 3. Modeling Methodology & Feature Scaling
* **Temporal Sorting & Split:** Sorted dataset chronologically (`Year`, `Month`, `Quarter`, `Week`) and performed an **80/20 chronological train/test split** (4,680 training rows, 1,170 testing rows) to prevent temporal data leakage.
* **Target Leakage Prevention:** Removed non-lagged columns (`Scarcity_Flag`) from feature matrix $X$, ensuring predictions depend strictly on historical indicators.
* **Feature Scaling (`StandardScaler`):** Features were standardized using `StandardScaler` fitted on the training set to prepare the dataset for distance- and gradient-based models (such as the baseline Logistic Regression).
* **Final Model Selection:** `BalancedRandomForestClassifier` was selected as the primary production model to address heavy class imbalance (~10% scarcity prevalence in testing data). While the pipeline includes standard scaling, decision tree ensembles are scale-invariant and trained directly on the raw feature set to preserve direct interpretability of feature importance values.

---

## 4. Key Evaluation Results & Decision Threshold
* **ROC-AUC Score:** **0.7034** (demonstrating strong model discrimination between scarcity and normal supply states).
* **Optimal Decision Threshold:** **0.40** (Selected to balance real-world risk management; catching shortages is far more critical than avoiding false alarms).
* **Scarcity Recall:** **60.9%** (Successfully detected 78 out of 128 scarcity events in the test set, compared to 0% with a standard unadjusted classifier).

---

## 5. Top Predictive Drivers (Feature Importances)
1. `Supply_Demand_Ratio_lag_1`
2. `Exchange_Rate_vs_USD`
3. `Crude_lag_1`
4. `Inflation_Rate (%)`

---

## 6. Repository Structure 
* `fuel_raw_data.csv` - Raw Global fuel price dataset from kaggle
* `Data_Prep_and_EDA.ipynb` - Data loading, cleaning, missing value handling, target engineering, and seasonal exploratory analysis.
* `Feature_Eng_and_Model.ipynb` - Chronological train/test split, model training, threshold tuning, and feature importance evaluation.
* `fuel_processed_data.csv` - Processed time-series dataset export.
* `fuel_scarcity_model.pkl` - Saved `BalancedRandomForestClassifier` model artifact.
* `scaler.pkl` — Standard scaler fitted on training features.

---

## 7. How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/Legendmbank/Fuel-scarcity-forecasting.git](https://github.com/Legendmbank/Fuel-scarcity-forecasting.git)
