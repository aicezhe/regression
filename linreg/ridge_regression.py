import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


conn = sqlite3.connect("housing.db")
pd.read_csv("Housing.csv").to_sql("houses", conn, if_exists="replace", index=False)

df = pd.read_sql("SELECT * FROM houses", conn)

print("=" * 60)
print("Dataset shape:", df.shape)
print("\nFirst rows:")
print(df.head(3).to_string())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:", df.isnull().sum().sum())


print("\n--- SQL Analysis ---")

print("\nPrice statistics:")
print(pd.read_sql("""
    SELECT
        COUNT(*)              AS total_properties,
        ROUND(AVG(price), 0)  AS avg_price,
        MIN(price)            AS min_price,
        MAX(price)            AS max_price
    FROM houses
""", conn).to_string(index=False))

print("\nAverage price by bedrooms:")
print(pd.read_sql("""
    SELECT
        bedrooms,
        COUNT(*)               AS total,
        ROUND(AVG(price), 0)   AS avg_price
    FROM houses
    GROUP BY bedrooms
    ORDER BY bedrooms
""", conn).to_string(index=False))

print("\nAverage price by air conditioning:")
print(pd.read_sql("""
    SELECT
        airconditioning,
        COUNT(*)               AS total,
        ROUND(AVG(price), 0)   AS avg_price
    FROM houses
    GROUP BY airconditioning
    ORDER BY avg_price DESC
""", conn).to_string(index=False))

print("\nAverage price by furnishing status:")
print(pd.read_sql("""
    SELECT
        furnishingstatus,
        COUNT(*)               AS total,
        ROUND(AVG(price), 0)   AS avg_price
    FROM houses
    GROUP BY furnishingstatus
    ORDER BY avg_price DESC
""", conn).to_string(index=False))

print("\nTop 5 most expensive properties:")
print(pd.read_sql("""
    SELECT price, area, bedrooms, bathrooms, airconditioning, furnishingstatus
    FROM houses
    ORDER BY price DESC
    LIMIT 5
""", conn).to_string(index=False))


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("EDA — Housing Price Dataset", fontsize=16, fontweight="bold")

axes[0, 0].hist(df["price"] / 1e6, bins=30, color="#4C72B0", edgecolor="white")
axes[0, 0].set_title("Price distribution")
axes[0, 0].set_xlabel("Price (millions)")
axes[0, 0].set_ylabel("Frequency")

axes[0, 1].scatter(df["area"], df["price"] / 1e6, alpha=0.5, color="#DD8452")
axes[0, 1].set_title("Price vs Area")
axes[0, 1].set_xlabel("Area (sq ft)")
axes[0, 1].set_ylabel("Price (millions)")

df_plot = df.copy()
df_plot["price_M"] = df_plot["price"] / 1e6
sns.boxplot(data=df_plot, x="bedrooms", y="price_M", ax=axes[0, 2], color="#4C72B0")
axes[0, 2].set_title("Price by bedrooms")
axes[0, 2].set_xlabel("Bedrooms")
axes[0, 2].set_ylabel("Price (millions)")

df_plot["AC"] = df_plot["airconditioning"].map({"yes": "Yes", "no": "No"})
sns.boxplot(data=df_plot, x="AC", y="price_M", ax=axes[1, 0], color="#DD8452")
axes[1, 0].set_title("Price: AC vs no AC")
axes[1, 0].set_xlabel("Air conditioning")
axes[1, 0].set_ylabel("Price (millions)")

sns.boxplot(data=df_plot, x="furnishingstatus", y="price_M", ax=axes[1, 1], color="#55A868")
axes[1, 1].set_title("Price by furnishing")
axes[1, 1].set_xlabel("Furnishing status")
axes[1, 1].set_ylabel("Price (millions)")

df_numeric = df.copy()
binary_cols = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    df_numeric[col] = (df_numeric[col] == "yes").astype(int)
df_numeric["furnishingstatus"] = df_numeric["furnishingstatus"].map(
    {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
)
corr_price = df_numeric.corr()["price"].drop("price").abs().sort_values(ascending=True)
axes[1, 2].barh(corr_price.index, corr_price.values, color="#55A868")
axes[1, 2].set_title("Feature correlation with price")
axes[1, 2].set_xlabel("|Pearson correlation|")

plt.tight_layout()
plt.savefig("eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] eda.png")


df = df_numeric.copy()
df["price_log"] = np.log(df["price"])

print("\n--- Preprocessing ---")
print("Binary encoding: yes=1, no=0")
print("Furnishing status: furnished=2, semi-furnished=1, unfurnished=0")
print("Target: log(price) to reduce skewness")


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
y = df["price_log"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n--- Split ---")
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


print("\n--- GridSearchCV: tuning alpha ---")

alpha_grid = {"alpha": [0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000]}
grid_search = GridSearchCV(
    Ridge(), param_grid=alpha_grid, cv=5, scoring="r2", n_jobs=-1
)
grid_search.fit(X_train_sc, y_train)

best_alpha = grid_search.best_params_["alpha"]
best_cv_r2 = grid_search.best_score_
print(f"Best alpha:  {best_alpha}")
print(f"Best CV R²: {best_cv_r2:.4f}")

cv_results = pd.DataFrame(grid_search.cv_results_)
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(cv_results["param_alpha"].astype(float),
            cv_results["mean_test_score"],
            marker="o", color="#4C72B0", linewidth=2)
ax.axvline(best_alpha, color="red", linestyle="--", label=f"Best α={best_alpha}")
ax.fill_between(cv_results["param_alpha"].astype(float),
                cv_results["mean_test_score"] - cv_results["std_test_score"],
                cv_results["mean_test_score"] + cv_results["std_test_score"],
                alpha=0.15, color="#4C72B0")
ax.set_xlabel("Alpha (log scale)")
ax.set_ylabel("CV R²")
ax.set_title("GridSearchCV — Ridge alpha tuning")
ax.legend()
plt.tight_layout()
plt.savefig("alpha_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] alpha_tuning.png")


model = Ridge(alpha=best_alpha)
model.fit(X_train_sc, y_train)

cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="r2")
print(f"\n--- Final Model (α={best_alpha}) ---")
print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


y_pred_log  = model.predict(X_test_sc)
y_pred      = np.exp(y_pred_log)
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


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Ridge Regression Results (α={best_alpha})", fontsize=14, fontweight="bold")

axes[0].scatter(y_test_orig / 1e6, y_pred / 1e6, alpha=0.6, color="#4C72B0")
mn = min(y_test_orig.min(), y_pred.min()) / 1e6
mx = max(y_test_orig.max(), y_pred.max()) / 1e6
axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect fit")
axes[0].set_xlabel("Actual price (millions)")
axes[0].set_ylabel("Predicted price (millions)")
axes[0].set_title("Predicted vs Actual")
axes[0].legend()
axes[0].text(0.05, 0.92, f"R² = {r2:.3f}", transform=axes[0].transAxes, fontsize=11)

residuals = y_test_orig.values - y_pred
axes[1].scatter(y_pred / 1e6, residuals / 1e6, alpha=0.6, color="#DD8452")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Predicted price (millions)")
axes[1].set_ylabel("Residual (millions)")
axes[1].set_title("Residuals")

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

print("\n Done!")
