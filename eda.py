import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv("Housing.csv")

print("=" * 60)
print("Dataset shape:", df.shape)
print("\nFirst rows:")
print(df.head(5).to_string())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe())



df_enc = df.copy()
binary_cols = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    df_enc[col] = (df_enc[col] == "yes").astype(int)
df_enc["furnishingstatus"] = df_enc["furnishingstatus"].map(
    {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
)



fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("EDA — Housing Price Dataset", fontsize=16, fontweight="bold")

# --- Price distribution ---
axes[0, 0].hist(df["price"] / 1e6, bins=30, color="#4C72B0", edgecolor="white")
axes[0, 0].set_title("Price distribution")
axes[0, 0].set_xlabel("Price (millions)")
axes[0, 0].set_ylabel("Frequency")

# --- Price vs Area ---
axes[0, 1].scatter(df["area"], df["price"] / 1e6, alpha=0.5, color="#DD8452")
axes[0, 1].set_title("Price vs Area")
axes[0, 1].set_xlabel("Area (sq ft)")
axes[0, 1].set_ylabel("Price (millions)")

# --- Price by bedrooms ---
df_plot = df.copy()
df_plot["price_M"] = df_plot["price"] / 1e6
sns.boxplot(data=df_plot, x="bedrooms", y="price_M", ax=axes[0, 2], color="#4C72B0")
axes[0, 2].set_title("Price by bedrooms")
axes[0, 2].set_xlabel("Bedrooms")
axes[0, 2].set_ylabel("Price (millions)")

# --- Price by air conditioning ---
df_plot["AC"] = df_plot["airconditioning"].map({"yes": "Yes", "no": "No"})
sns.boxplot(data=df_plot, x="AC", y="price_M", ax=axes[1, 0], color="#DD8452")
axes[1, 0].set_title("Price: AC vs no AC")
axes[1, 0].set_xlabel("Air conditioning")
axes[1, 0].set_ylabel("Price (millions)")

# --- Price by furnishing status ---
sns.boxplot(data=df_plot, x="furnishingstatus", y="price_M", ax=axes[1, 1], color="#55A868")
axes[1, 1].set_title("Price by furnishing")
axes[1, 1].set_xlabel("Furnishing status")
axes[1, 1].set_ylabel("Price (millions)")

# --- Correlation with price ---
corr_price = df_enc.corr()["price"].drop("price").abs().sort_values(ascending=True)
axes[1, 2].barh(corr_price.index, corr_price.values, color="#55A868")
axes[1, 2].set_title("Feature correlation with price")
axes[1, 2].set_xlabel("|Pearson correlation|")

plt.tight_layout()
plt.savefig("eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] eda.png")



fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(df_enc.corr(), dtype=bool))
sns.heatmap(df_enc.corr(), mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Correlation heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] eda_heatmap.png")

print("\n EDA complete!")
