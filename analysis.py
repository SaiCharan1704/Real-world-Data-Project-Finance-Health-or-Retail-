"""
╔══════════════════════════════════════════════════════════════════╗
║     Real-World Data Project — Finance / Health / Retail          ║
║     End-to-End Domain-Specific Data Analysis & Prediction        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Dark Theme ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",   "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",      "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",       "grid.color": "#21262d",
    "grid.linestyle": "--",        "font.family": "monospace",
})
C = {"blue":"#58a6ff","green":"#3fb950","red":"#f78166",
     "orange":"#ffa657","purple":"#d2a8ff","yellow":"#e3b341"}

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 1 — FINANCE: Stock Price Analysis & Trend Prediction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  📈 DOMAIN 1: FINANCE — Stock Price Analysis")
print("═"*60)

dates = pd.date_range("2023-01-01", periods=365, freq="D")
trend = np.linspace(100, 165, 365)
noise = np.random.normal(0, 4, 365)
seasonal = 8 * np.sin(np.linspace(0, 4*np.pi, 365))
close = trend + noise + seasonal

finance_df = pd.DataFrame({
    "Date": dates, "Close": close,
    "Open":  close + np.random.normal(0, 1.5, 365),
    "High":  close + np.abs(np.random.normal(2, 1, 365)),
    "Low":   close - np.abs(np.random.normal(2, 1, 365)),
    "Volume": np.random.randint(1_000_000, 10_000_000, 365),
})
finance_df["MA_7"]  = finance_df["Close"].rolling(7).mean()
finance_df["MA_30"] = finance_df["Close"].rolling(30).mean()
finance_df["Daily_Return"] = finance_df["Close"].pct_change() * 100
finance_df["Volatility_30"] = finance_df["Daily_Return"].rolling(30).std()

print(f"  Records      : {len(finance_df)}")
print(f"  Start Price  : ${finance_df['Close'].iloc[0]:.2f}")
print(f"  End Price    : ${finance_df['Close'].iloc[-1]:.2f}")
print(f"  Total Return : {((finance_df['Close'].iloc[-1]/finance_df['Close'].iloc[0])-1)*100:.1f}%")
print(f"  Avg Daily Vol: {finance_df['Volume'].mean():,.0f}")
print(f"  Max Drawdown : {(finance_df['Close'].min()-finance_df['Close'].max())/finance_df['Close'].max()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 2 — HEALTH: Patient Data Analysis & Risk Prediction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  🏥 DOMAIN 2: HEALTH — Patient Risk Analysis")
print("═"*60)

n = 800
health_df = pd.DataFrame({
    "Age":          np.random.randint(25, 80, n),
    "BMI":          np.random.normal(27.5, 5.5, n).clip(16, 45),
    "Blood_Pressure":np.random.normal(120, 18, n).clip(80, 190),
    "Cholesterol":  np.random.normal(200, 35, n).clip(120, 320),
    "Glucose":      np.random.normal(95, 22, n).clip(60, 200),
    "Smoker":       np.random.choice([0,1], n, p=[0.73, 0.27]),
    "Exercise_Hrs": np.random.exponential(3, n).clip(0, 14),
})
risk_score = (
    (health_df["Age"] > 55).astype(int) * 2 +
    (health_df["BMI"] > 30).astype(int) * 2 +
    (health_df["Blood_Pressure"] > 130).astype(int) * 2 +
    (health_df["Cholesterol"] > 240).astype(int) +
    health_df["Smoker"] * 3 +
    (health_df["Exercise_Hrs"] < 2).astype(int)
)
health_df["Risk_Level"] = pd.cut(risk_score, bins=[-1,2,5,10],
                                  labels=["Low","Medium","High"])

X_h = health_df.drop(columns=["Risk_Level"])
y_h = LabelEncoder().fit_transform(health_df["Risk_Level"])
X_tr, X_te, y_tr, y_te = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
acc = accuracy_score(y_te, clf.predict(X_te))

print(f"  Patients     : {n}")
print(f"  High Risk    : {(health_df['Risk_Level']=='High').sum()} ({(health_df['Risk_Level']=='High').mean()*100:.1f}%)")
print(f"  Medium Risk  : {(health_df['Risk_Level']=='Medium').sum()}")
print(f"  Low Risk     : {(health_df['Risk_Level']=='Low').sum()}")
print(f"  Model (RF)   : Accuracy = {acc:.4f}")
print(f"  Avg BMI      : {health_df['BMI'].mean():.1f}")
print(f"  Smokers      : {health_df['Smoker'].mean()*100:.1f}%")

feat_imp = pd.Series(clf.feature_importances_, index=X_h.columns).sort_values(ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 3 — RETAIL: Sales Data Analysis & Revenue Prediction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  🛒 DOMAIN 3: RETAIL — Sales Analysis & Forecasting")
print("═"*60)

months = pd.date_range("2022-01-01", periods=24, freq="ME")
categories = ["Electronics","Clothing","Groceries","Furniture","Beauty"]
retail_rows = []
for cat in categories:
    base = {"Electronics":85000,"Clothing":52000,"Groceries":38000,
            "Furniture":62000,"Beauty":29000}[cat]
    for i, m in enumerate(months):
        seasonal_mult = 1 + 0.3*np.sin((i+2)*np.pi/6)
        sales = base * seasonal_mult * np.random.uniform(0.88, 1.12)
        retail_rows.append({"Month":m,"Category":cat,"Sales":sales,
                             "Units":int(sales/np.random.uniform(25,150)),
                             "Discount_Pct":np.random.uniform(0,0.35)})
retail_df = pd.DataFrame(retail_rows)
retail_df["Profit"] = retail_df["Sales"] * (0.28 - retail_df["Discount_Pct"]/3)

total_rev = retail_df["Sales"].sum()
best_cat  = retail_df.groupby("Category")["Sales"].sum().idxmax()
print(f"  Months       : 24 (2 years)")
print(f"  Total Revenue: ${total_rev:,.0f}")
print(f"  Best Category: {best_cat}")
print(f"  Total Profit : ${retail_df['Profit'].sum():,.0f}")
print(f"  Avg Monthly  : ${retail_df.groupby('Month')['Sales'].sum().mean():,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Plot 1: Finance Dashboard ─────────────────────────────────────────────────
fig = plt.figure(figsize=(18,12), facecolor="#0d1117")
fig.suptitle("📈 Finance Domain — Stock Price Analysis", fontsize=17,
             color=C["blue"], fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0,:2])
ax1.plot(finance_df["Date"], finance_df["Close"], color=C["blue"], lw=1.2, alpha=0.8, label="Close")
ax1.plot(finance_df["Date"], finance_df["MA_7"],  color=C["orange"], lw=1.5, label="MA-7")
ax1.plot(finance_df["Date"], finance_df["MA_30"], color=C["green"],  lw=1.8, label="MA-30")
ax1.fill_between(finance_df["Date"], finance_df["Close"], alpha=0.1, color=C["blue"])
ax1.set_title("Stock Price with Moving Averages", color=C["blue"]); ax1.legend(fontsize=9)
ax1.set_xlabel("Date"); ax1.set_ylabel("Price ($)")

ax2 = fig.add_subplot(gs[0,2])
ax2.bar(finance_df["Date"], finance_df["Volume"]/1e6, color=C["purple"], alpha=0.7, width=1)
ax2.set_title("Daily Trading Volume", color=C["purple"])
ax2.set_xlabel("Date"); ax2.set_ylabel("Volume (M)")

ax3 = fig.add_subplot(gs[1,0])
returns = finance_df["Daily_Return"].dropna()
ax3.hist(returns, bins=40, color=C["green"], alpha=0.8, edgecolor="#0d1117")
ax3.axvline(returns.mean(), color=C["red"], lw=2, label=f"Mean={returns.mean():.2f}%")
ax3.set_title("Daily Returns Distribution", color=C["green"])
ax3.set_xlabel("Return (%)"); ax3.legend(fontsize=9)

ax4 = fig.add_subplot(gs[1,1])
ax4.plot(finance_df["Date"], finance_df["Volatility_30"], color=C["orange"], lw=1.5)
ax4.fill_between(finance_df["Date"], finance_df["Volatility_30"], alpha=0.2, color=C["orange"])
ax4.set_title("30-Day Rolling Volatility", color=C["orange"])
ax4.set_xlabel("Date"); ax4.set_ylabel("Std Dev (%)")

ax5 = fig.add_subplot(gs[1,2])
monthly = finance_df.set_index("Date")["Close"].resample("ME").last()
colors_bar = [C["green"] if v > 0 else C["red"]
              for v in monthly.pct_change().fillna(0)]
ax5.bar(range(len(monthly)), monthly.pct_change().fillna(0)*100, color=colors_bar)
ax5.set_title("Monthly Return %", color=C["yellow"])
ax5.set_xlabel("Month"); ax5.set_ylabel("Return (%)")
ax5.axhline(0, color="#30363d", lw=1)

plt.savefig("finance_dashboard.png", dpi=150, bbox_inches="tight"); plt.close()

# ── Plot 2: Health Dashboard ──────────────────────────────────────────────────
fig = plt.figure(figsize=(18,12), facecolor="#0d1117")
fig.suptitle("🏥 Health Domain — Patient Risk Analysis", fontsize=17,
             color=C["green"], fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2,3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0,0])
risk_counts = health_df["Risk_Level"].value_counts()
wedge_c = [C["red"],C["orange"],C["green"]]
ax1.pie(risk_counts, labels=risk_counts.index, colors=wedge_c,
        autopct="%1.1f%%", textprops={"color":"#c9d1d9","fontsize":10}, startangle=90)
ax1.set_title("Patient Risk Distribution", color=C["red"])

ax2 = fig.add_subplot(gs[0,1])
for lvl, col in zip(["Low","Medium","High"],[C["green"],C["orange"],C["red"]]):
    subset = health_df[health_df["Risk_Level"]==lvl]["BMI"]
    ax2.hist(subset, bins=25, alpha=0.65, color=col, label=lvl, edgecolor="#0d1117")
ax2.set_title("BMI by Risk Level", color=C["blue"])
ax2.set_xlabel("BMI"); ax2.set_ylabel("Count"); ax2.legend(fontsize=9)

ax3 = fig.add_subplot(gs[0,2])
ax3.barh(feat_imp.index, feat_imp.values,
         color=[C["blue"],C["green"],C["orange"],C["purple"],C["red"],C["yellow"],C["green"]][:len(feat_imp)])
ax3.set_title("Feature Importance (RF)", color=C["purple"])
ax3.set_xlabel("Importance")

ax4 = fig.add_subplot(gs[1,0])
sc = ax4.scatter(health_df["Age"], health_df["Blood_Pressure"],
                 c=health_df["BMI"], cmap="RdYlGn_r", alpha=0.5, s=18)
plt.colorbar(sc, ax=ax4, label="BMI")
ax4.set_title("Age vs Blood Pressure", color=C["blue"])
ax4.set_xlabel("Age"); ax4.set_ylabel("Blood Pressure")

ax5 = fig.add_subplot(gs[1,1])
ex_bins = [0,1,3,6,9,14]
labels = ["<1","1-3","3-6","6-9","9+"]
health_df["Ex_Group"] = pd.cut(health_df["Exercise_Hrs"], bins=ex_bins, labels=labels)
ex_risk = health_df.groupby("Ex_Group", observed=True)["Risk_Level"].apply(
    lambda x: (x=="High").mean()*100)
ax5.bar(ex_risk.index, ex_risk.values, color=C["orange"], alpha=0.85)
ax5.set_title("High-Risk % by Exercise", color=C["orange"])
ax5.set_xlabel("Exercise Hrs/Week"); ax5.set_ylabel("High Risk %")

ax6 = fig.add_subplot(gs[1,2])
corr_h = health_df[["Age","BMI","Blood_Pressure","Cholesterol","Glucose","Exercise_Hrs"]].corr()
sns.heatmap(corr_h, ax=ax6, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, linecolor="#0d1117", annot_kws={"size":8,"color":"white"},
            cbar_kws={"shrink":0.8})
ax6.set_title("Health Feature Correlations", color=C["green"])

plt.savefig("health_dashboard.png", dpi=150, bbox_inches="tight"); plt.close()

# ── Plot 3: Retail Dashboard ──────────────────────────────────────────────────
fig = plt.figure(figsize=(18,12), facecolor="#0d1117")
fig.suptitle("🛒 Retail Domain — Sales & Revenue Analysis", fontsize=17,
             color=C["orange"], fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2,3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0,:2])
cat_colors = [C["blue"],C["green"],C["orange"],C["purple"],C["red"]]
for cat, col in zip(categories, cat_colors):
    sub = retail_df[retail_df["Category"]==cat].sort_values("Month")
    ax1.plot(sub["Month"], sub["Sales"]/1000, marker="o", ms=4, lw=2, label=cat, color=col)
ax1.set_title("Monthly Sales by Category (2022–2024)", color=C["orange"])
ax1.set_xlabel("Month"); ax1.set_ylabel("Sales ($K)"); ax1.legend(fontsize=9)

ax2 = fig.add_subplot(gs[0,2])
cat_totals = retail_df.groupby("Category")["Sales"].sum().sort_values()
ax2.barh(cat_totals.index, cat_totals.values/1e6, color=cat_colors[::-1], alpha=0.85)
ax2.set_title("Total Revenue by Category", color=C["blue"])
ax2.set_xlabel("Revenue ($M)")

ax3 = fig.add_subplot(gs[1,0])
monthly_total = retail_df.groupby("Month")["Sales"].sum()
ax3.fill_between(monthly_total.index, monthly_total/1000, alpha=0.3, color=C["green"])
ax3.plot(monthly_total.index, monthly_total/1000, color=C["green"], lw=2)
ax3.set_title("Total Monthly Revenue", color=C["green"])
ax3.set_xlabel("Month"); ax3.set_ylabel("Sales ($K)")

ax4 = fig.add_subplot(gs[1,1])
ax4.scatter(retail_df["Discount_Pct"]*100, retail_df["Profit"]/1000,
            c=[cat_colors[categories.index(c)] for c in retail_df["Category"]],
            alpha=0.5, s=30)
ax4.set_title("Discount % vs Profit", color=C["purple"])
ax4.set_xlabel("Discount (%)"); ax4.set_ylabel("Profit ($K)")

ax5 = fig.add_subplot(gs[1,2])
profit_by_cat = retail_df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
ax5.bar(profit_by_cat.index, profit_by_cat/1e6, color=cat_colors, alpha=0.85)
ax5.set_title("Total Profit by Category", color=C["yellow"])
ax5.set_xlabel("Category"); ax5.set_ylabel("Profit ($M)")
plt.xticks(rotation=15)

plt.savefig("retail_dashboard.png", dpi=150, bbox_inches="tight"); plt.close()

print("\n" + "═"*60)
print("✅ All dashboards saved:")
print("   • finance_dashboard.png")
print("   • health_dashboard.png")
print("   • retail_dashboard.png")
print("═"*60)
