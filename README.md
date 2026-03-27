# Housing Price Prediction

Predicting residential property prices using supervised machine learning regression models, based on a structured dataset of 545 properties.

---

## Dataset

Sourced from Kaggle as a practice dataset. 545 records, 13 features, no missing values.

| Type | Features |
|---|---|
| Numerical | area, bedrooms, bathrooms, stories, parking |
| Binary (yes/no) | mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea |
| Categorical | furnishingstatus (furnished / semi-furnished / unfurnished) |
| Target | price |

---

## Model Selection

Analyzing the data revealed sufficiently linear dependencies — a natural fit for linear regression. Rather than plain linear regression, **Ridge (L2)** was chosen: it adds a penalty for large coefficients, stabilizing predictions when features are correlated (e.g. `area` and `bedrooms`). This distributes weight more evenly and generalizes better to unseen data.

**Random Forest** was selected as the comparison model. While the data is largely linear, some features — air conditioning, preferred area location — influence price non-linearly and through interactions. Random Forest captures these effects well, handles ~500 rows without overfitting, and provides built-in feature importance for interpretation.

---

## Exploratory Data Analysis

**Target variable.** Prices range from 1.75M to 13.3M with a mean of 4.77M. The distribution is right-skewed — a small number of expensive properties pull the average upward. A log transformation was applied before Ridge training to bring the distribution closer to normality.

**Strongest predictors.** `area` and `bathrooms` show the highest correlation with price (~0.53), followed by `airconditioning`, `stories`, and `parking`. Notably, `bedrooms` correlates weaker than expected — bedroom count alone doesn't capture property quality. `hotwaterheating` and `basement` show minimal correlation (<0.2) and contribute little to predictions.

**Categorical insights.** Properties with air conditioning are priced ~1.5× higher at the median. Preferred area location similarly corresponds to a significant price premium. Furnishing status shows a clear but moderate gradient: furnished > semi-furnished > unfurnished — visible across all price ranges including outliers.

**Bedroom pattern.** Median price grows from 1 to 4 bedrooms, then plateaus and slightly drops at 5–6. Properties with more bedrooms also show much wider price spread and more outliers, meaning the model's uncertainty increases for larger homes — consistent with `bedrooms` being a weak linear predictor.

---

## Preprocessing

Binary features were encoded as 0/1. Furnishing status was ordinally encoded (unfurnished=0, semi-furnished=1, furnished=2) reflecting a natural quality progression. For Ridge, a log transformation was applied to `price` and all features were standardized with `StandardScaler` — required so the L2 penalty applies evenly across coefficients of different scales. Random Forest required neither transformation nor scaling, as tree-based models are scale-invariant.

**Data loading via SQL.** In the Ridge pipeline, data is loaded into a SQLite database and queried using SQL rather than read directly from CSV. In real company environments, data rarely lives in flat files — it is stored in relational databases and accessed through SQL queries. By integrating SQLite and `pd.read_sql()`, this project reflects that workflow and demonstrates basic SQL proficiency alongside Python.

---

## Feature Engineering

Three features were constructed to capture relationships that raw variables miss:

**`area_per_room`** — total area divided by bedrooms + bathrooms. A 6000 sq ft house with 2 rooms is fundamentally different from the same size with 8 rooms.

**`amenities_score`** — sum of all six binary amenity features (0–6). Overall comfort level matters as much as any single amenity — a house scoring 5 is likely in a different price bracket than one scoring 1.

**`bath_bed_ratio`** — bathrooms relative to bedrooms. A higher ratio signals a more premium property, as extra bathrooms beyond bedrooms are a common luxury marker.

---

## Results

### Ridge Regression

| Metric | Value |
|---|---|
| R² | 0.6494 |
| MAE | 977,497 |
| RMSE | 1,331,299 |
| MAPE | 20.99% |

The model explains 65% of price variance — a reasonable baseline for 545 records with no location data. The gap between RMSE and MAE indicates larger errors on high-end outliers, confirmed by the residuals plot which shows mild heteroscedasticity at higher predicted prices. Feature coefficients identify `area`, `bathrooms`, and `airconditioning` as the strongest positive predictors — consistent with EDA findings.

### Random Forest

| Metric | Value |
|---|---|
| R² | 0.6666 |
| MAE | 935,038 |
| RMSE | 1,298,143 |
| MAPE | 19.97% |

Random Forest outperforms Ridge across all metrics. Feature importance confirms `area` as dominant (0.44), followed by `bathrooms` (0.11) and `amenities_score` (0.10). `mainroad`, `guestroom`, and `basement` contribute minimally (<0.01). 61.5% of predictions fall within 20% of the true price, and 76.1% within 30%.

### Comparison

| Metric | Ridge | Random Forest |
|---|---|---|
| R² | 0.6494 | **0.6666** |
| MAE | 977,497 | **935,038** |
| MAPE | 20.99% | **19.97%** |

The improvement is moderate — confirming that the data is largely linear, but non-linear interactions (captured by Random Forest) do add meaningful predictive value. The remaining unexplained variance (~33%) is primarily attributable to the absence of location data, which is consistently the strongest driver of real estate prices in practice.
