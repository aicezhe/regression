import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("Housing.csv")
print("=" * 60)
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())


binary_cols = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    df[col] = (df[col] == "yes").astype(int)


df["furnishingstatus"] = df["furnishingstatus"].map(
    {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
)



print("\n--- Preprocessing done ---")
print("Binary encoding: yes=1, no=0")
print("Furnishing: furnished=2, semi-furnished=1, unfurnished=0")
print("No log-transform or scaling needed for Random Forest")



df["area_per_room"]   = df["area"] / (df["bedrooms"] + df["bathrooms"])
df["amenities_score"] = (df["mainroad"] + df["guestroom"] + df["basement"] +
                         df["hotwaterheating"] + df["airconditioning"] + df["prefarea"])
df["bath_bed_ratio"]  = df["bathrooms"] / df["bedrooms"]

print("\n--- Feature Engineering ---")
print("New features: area_per_room, amenities_score, bath_bed_ratio")



FEATURES = ["area", "bedrooms", "bathrooms", "stories", "mainroad",
            "guestroom", "basement", "hotwaterheating", "airconditioning",
            "parking", "prefarea", "furnishingstatus",
            "area_per_room", "amenities_score", "bath_bed_ratio"]

X = df[FEATURES]
y = df["price"]  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n--- Split ---")
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")



print("\n--- GridSearchCV: tuning hyperparameters ---")

param_grid = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [None, 10, 20],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_params  = grid_search.best_params_
best_cv_r2   = grid_search.best_score_
print(f"Best params: {best_params}")
print(f"Best CV R²: {best_cv_r2:.4f}")



model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
print(f"\n--- Final Model ---")
print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")



y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n--- Test Set Metrics ---")
print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"MAPE: {mape:.2f}%")



fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Random Forest Results", fontsize=14, fontweight="bold")


axes[0].scatter(y_test / 1e6, y_pred / 1e6, alpha=0.6, color="#4C72B0")
mn = min(y_test.min(), y_pred.min()) / 1e6
mx = max(y_test.max(), y_pred.max()) / 1e6
axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect fit")
axes[0].set_xlabel("Actual price (millions)")
axes[0].set_ylabel("Predicted price (millions)")
axes[0].set_title("Predicted vs Actual")
axes[0].legend()
axes[0].text(0.05, 0.92, f"R² = {r2:.3f}", transform=axes[0].transAxes, fontsize=11)


residuals = y_test.values - y_pred
axes[1].scatter(y_pred / 1e6, residuals / 1e6, alpha=0.6, color="#DD8452")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Predicted price (millions)")
axes[1].set_ylabel("Residual (millions)")
axes[1].set_title("Residuals")


feat_imp = pd.Series(
    model.feature_importances_, index=FEATURES
).sort_values(ascending=True)

colors = ["#4C72B0" if v < feat_imp.median() else "#2ecc71" for v in feat_imp.values]
axes[2].barh(feat_imp.index, feat_imp.values, color=colors)
axes[2].set_title("Feature importance")
axes[2].set_xlabel("Importance score")

plt.tight_layout()
plt.savefig("rf_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] rf_results.png")



print("\n--- Comparison: Ridge vs Random Forest ---")
print(f"{'Metric':<10} {'Ridge':>10} {'Random Forest':>15}")
print("-" * 37)
print(f"{'R²':<10} {'0.6494':>10} {r2:>15.4f}")
print(f"{'MAPE':<10} {'20.99%':>10} {mape:>14.2f}%")
print(f"{'MAE':<10} {'977,497':>10} {mae:>15,.0f}")



print("\n--- Results Analysis ---")


if r2 >= 0.75:
    r2_comment = "strong — model explains most of the variance"
elif r2 >= 0.60:
    r2_comment = "moderate — reasonable baseline for this dataset size"
else:
    r2_comment = "weak — model struggles to explain variance"
print(f"\nR² = {r2:.4f} → {r2_comment}")


if mape <= 15:
    mape_comment = "good — predictions within 15% on average"
elif mape <= 25:
    mape_comment = "acceptable — typical for small real estate datasets"
else:
    mape_comment = "high — large average prediction error"
print(f"MAPE = {mape:.2f}% → {mape_comment}")


gap = rmse - mae
print(f"\nMAE  = {mae:,.0f}")
print(f"RMSE = {rmse:,.0f}")
print(f"Gap  = {gap:,.0f}")
if gap > 300_000:
    print("→ Large gap: model makes significant errors on some outlier properties")
else:
    print("→ Small gap: errors are relatively consistent across all properties")


print("\n--- Feature Importance ---")
top3 = feat_imp.sort_values(ascending=False).head(3)
bottom3 = feat_imp.sort_values(ascending=True).head(3)
print("Top 3 most important:")
for feat, score in top3.items():
    print(f"  {feat:<22} {score:.4f}")
print("Least important:")
for feat, score in bottom3.items():
    print(f"  {feat:<22} {score:.4f}")


within_10 = np.mean(np.abs((y_test.values - y_pred) / y_test.values) < 0.10) * 100
within_20 = np.mean(np.abs((y_test.values - y_pred) / y_test.values) < 0.20) * 100
within_30 = np.mean(np.abs((y_test.values - y_pred) / y_test.values) < 0.30) * 100
print(f"\n--- Prediction Accuracy Breakdown ---")
print(f"Within 10% of true price: {within_10:.1f}% of samples")
print(f"Within 20% of true price: {within_20:.1f}% of samples")
print(f"Within 30% of true price: {within_30:.1f}% of samples")

print("\n Done!")
