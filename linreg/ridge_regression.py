import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("Housing.csv")

print("=" * 60)
print("Dataset shape:", df.shape)
print("\nFirst rows:")
print(df.head(3).to_string())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:", df.isnull().sum().sum())



fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("EDA — Housing Price Dataset", fontsize=16, fontweight="bold")


axes[0, 0].hist(df["price"] / 1e6, bins=30, color="#4C72B0", edgecolor="white")
axes[0, 0].set_title("Price distribution")
axes[0, 0].set_xlabel("Price (millions)")
axes[0, 0].set_ylabel("Frequency")

# Price vs Area
axes[0, 1].scatter(df["area"], df["price"] / 1e6, alpha=0.5, color="#DD8452")
axes[0, 1].set_title("Price vs Area")
axes[0, 1].set_xlabel("Area (sq ft)")
axes[0, 1].set_ylabel("Price (millions)")

# Price by number of bedrooms
df.boxplot(column="price", by="bedrooms", ax=axes[0, 2])
axes[0, 2].set_title("Price by bedrooms")
axes[0, 2].set_xlabel("Bedrooms")
axes[0, 2].set_ylabel("Price")
plt.sca(axes[0, 2])
plt.title("Price by bedrooms")

# Price by air conditioning
df_temp = df.copy()
df_temp["airconditioning_label"] = df_temp["airconditioning"].map({"yes": "Yes", "no": "No"})
df_temp.boxplot(column="price", by="airconditioning_label", ax=axes[1, 0])
axes[1, 0].set_title("Price: AC vs no AC")
axes[1, 0].set_xlabel("Air conditioning")
axes[1, 0].set_ylabel("Price")
plt.sca(axes[1, 0])
plt.title("Price: AC vs no AC")

# Price by furnishing status
df_temp.boxplot(column="price", by="furnishingstatus", ax=axes[1, 1])
axes[1, 1].set_title("Price by furnishing")
axes[1, 1].set_xlabel("Furnishing status")
axes[1, 1].set_ylabel("Price")
plt.sca(axes[1, 1])
plt.title("Price by furnishing")

# Correlation with price (numeric only)
df_numeric = df.copy()
binary_cols = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    df_numeric[col] = (df_numeric[col] == "yes").astype(int)
furnishing_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
df_numeric["furnishingstatus"] = df_numeric["furnishingstatus"].map(furnishing_map)

corr_price = df_numeric.corr()["price"].drop("price").abs().sort_values(ascending=True)
axes[1, 2].barh(corr_price.index, corr_price.values, color="#55A868")
axes[1, 2].set_title("Feature correlation with price")
axes[1, 2].set_xlabel("|Pearson correlation|")

plt.tight_layout()
plt.savefig("eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] eda.png")


# =============================================================
# 3. PREPROCESSING
# =============================================================

df = df_numeric.copy()

# Log-transform target to reduce skewness
df["price_log"] = np.log(df["price"])

print("\n--- Preprocessing ---")
print("Binary encoding: yes=1, no=0")
print("Furnishing status: furnished=2, semi-furnished=1, unfurnished=0")
print("Target: log(price) to reduce skewness")


# =============================================================
# 4. FEATURE ENGINEERING
# =============================================================

# Price per square foot (interaction feature)
df["area_per_room"] = df["area"] / (df["bedrooms"] + df["bathrooms"])

# Total amenities score
df["amenities_score"] = (df["mainroad"] + df["guestroom"] + df["basement"] +
                         df["hotwaterheating"] + df["airconditioning"] + df["prefarea"])

# Bathroom to bedroom ratio
df["bath_bed_ratio"] = df["bathrooms"] / df["bedrooms"]

print("\n--- Feature Engineering ---")
print("New features: area_per_room, amenities_score, bath_bed_ratio")


# =============================================================
# 5. TRAIN / TEST SPLIT
# =============================================================

FEATURES = ["area", "bedrooms", "bathrooms", "stories", "mainroad",
            "guestroom", "basement", "hotwaterheating", "airconditioning",
            "parking", "prefarea", "furnishingstatus",
            "area_per_room", "amenities_score", "bath_bed_ratio"]

X = df[FEATURES]
y = df["price_log"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n--- Split ---")
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Scale features (required for Ridge)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# =============================================================
# 6. HYPERPARAMETER TUNING — GridSearchCV for alpha
# =============================================================

print("\n--- GridSearchCV: tuning alpha ---")

alpha_grid = {"alpha": [0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000]}

grid_search = GridSearchCV(
    Ridge(),
    param_grid=alpha_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
grid_search.fit(X_train_sc, y_train)

best_alpha = grid_search.best_params_["alpha"]
best_cv_r2 = grid_search.best_score_

print(f"Best alpha:  {best_alpha}")
print(f"Best CV R²: {best_cv_r2:.4f}")

# Plot alpha search results
cv_results = pd.DataFrame(grid_search.cv_results_)
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(
    cv_results["param_alpha"].astype(float),
    cv_results["mean_test_score"],
    marker="o", color="#4C72B0", linewidth=2
)
ax.axvline(best_alpha, color="red", linestyle="--", label=f"Best α={best_alpha}")
ax.fill_between(
    cv_results["param_alpha"].astype(float),
    cv_results["mean_test_score"] - cv_results["std_test_score"],
    cv_results["mean_test_score"] + cv_results["std_test_score"],
    alpha=0.15, color="#4C72B0"
)
ax.set_xlabel("Alpha (log scale)")
ax.set_ylabel("CV R²")
ax.set_title("GridSearchCV — Ridge alpha tuning")
ax.legend()
plt.tight_layout()
plt.savefig("alpha_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] alpha_tuning.png")


# =============================================================
# 7. FINAL MODEL TRAINING
# =============================================================

model = Ridge(alpha=best_alpha)
model.fit(X_train_sc, y_train)

# Cross-validation score on full train set
cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="r2")
print(f"\n--- Final Model (α={best_alpha}) ---")
print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# =============================================================
# 8. EVALUATION ON TEST SET
# =============================================================

y_pred_log = model.predict(X_test_sc)

# Convert back from log scale
y_pred = np.exp(y_pred_log)
y_test_orig = np.exp(y_test)

mae  = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2   = r2_score(y_test_orig, y_pred)
mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100

print(f"\n--- Test Set Metrics ---")
print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"MAPE: {mape:.2f}%")


# =============================================================
# 9. RESULTS VISUALIZATION
# =============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Ridge Regression Results (α={best_alpha})", fontsize=14, fontweight="bold")

# Predicted vs Actual
axes[0].scatter(y_test_orig / 1e6, y_pred / 1e6, alpha=0.6, color="#4C72B0")
mn = min(y_test_orig.min(), y_pred.min()) / 1e6
mx = max(y_test_orig.max(), y_pred.max()) / 1e6
axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect fit")
axes[0].set_xlabel("Actual price (millions)")
axes[0].set_ylabel("Predicted price (millions)")
axes[0].set_title("Predicted vs Actual")
axes[0].legend()
axes[0].text(0.05, 0.92, f"R² = {r2:.3f}", transform=axes[0].transAxes,
             fontsize=11, color="#333")

# Residuals plot
residuals = y_test_orig.values - y_pred
axes[1].scatter(y_pred / 1e6, residuals / 1e6, alpha=0.6, color="#DD8452")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Predicted price (millions)")
axes[1].set_ylabel("Residual (millions)")
axes[1].set_title("Residuals")

# Feature coefficients
coef_df = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=True)

colors = ["#DD8452" if c > 0 else "#4C72B0" for c in coef_df["coefficient"]]
axes[2].barh(coef_df["feature"], coef_df["coefficient"], color=colors)
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].set_title("Feature coefficients")
axes[2].set_xlabel("Coefficient value")

plt.tight_layout()
plt.savefig("ridge_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] ridge_results.png")

print("\n Done! Files saved: eda.png, alpha_tuning.png, ridge_results.png")
