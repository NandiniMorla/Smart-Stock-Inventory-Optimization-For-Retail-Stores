# Milestone 3: Inventory Optimization
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# STREAMLIT CONFIG 
st.set_page_config(page_title="Inventory Optimization Dashboard", layout="wide")

# CUSTOM INLINE CSS 
st.markdown("""
<style>
    /* Global page background */
    body {
        background-color: #f6f9fc;
        color: #1c1c1c;
        font-family: 'Poppins', sans-serif;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #004e92, #000428);
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #fff;
    }
    [data-testid="stSidebar"] .stSelectbox label, .stSlider label {
        color: #c9d6e0 !important;
        font-weight: 600;
    }
    /* Headings */
    h1, h2, h3 {
        color: #003366;
        font-weight: 700;
    }
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        color: #004e92;
        font-weight: 700;
        font-size: 22px;
    }
    div[data-testid="stMetricLabel"] {
        color: #002b5b;
        font-weight: 600;
    }
    /* Add card-like effect */
    .block-container {
        padding-top: 1rem;
    }
    /* Subheaders */
    .stSubheader {
        color: #004e92 !important;
    }
    /* Chart container */
    .stPlotlyChart, .stPyplot {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }
    /* Metric container hover effect */
    div[data-testid="column"]:hover {
        transform: scale(1.02);
        transition: all 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# PAGE TITLE 
st.title("📦 Inventory Optimization Dashboard — Milestone 3")
# st.markdown("### 💼 Smarter Inventory Management with Seasonal & Demand Insights")

# LOAD DATA 
forecast_df = pd.read_csv("forecast_results.csv")
sales_df = pd.read_csv("cleaned_retail_sales.csv")

# Detect correct product column
if "Product ID" in forecast_df.columns:
    product_col = "Product ID"
elif "product_id" in forecast_df.columns:
    product_col = "product_id"
elif "product" in forecast_df.columns:
    product_col = "product"
else:
    st.error("❌ Could not find product column in forecast_results.csv")
    st.stop()

if "forecast_best" not in forecast_df.columns:
    st.error("❌ Column 'forecast_best' missing in forecast_results.csv")
    st.stop()

# Combine product & category for display
if "category" in forecast_df.columns:
    forecast_df["Product_Display"] = forecast_df[product_col].astype(str) + " - " + forecast_df["category"].astype(str)
else:
    forecast_df["Product_Display"] = forecast_df[product_col].astype(str)

# SIDEBAR 
st.sidebar.title("⚙️ Configuration Panel")
selected_product_display = st.sidebar.selectbox("🛍️ Select Product", forecast_df["Product_Display"].unique())
selected_product = forecast_df.loc[forecast_df["Product_Display"] == selected_product_display, product_col].iloc[0]

st.sidebar.markdown("### 📊 Inventory Parameters")
lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 10, 200, 50)
holding_cost = st.sidebar.slider("Holding Cost ($/unit)", 1, 20, 2)
service_levels = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
z = service_levels[st.sidebar.selectbox("Service Level", list(service_levels.keys()), 1)]

# INVENTORY CALCULATIONS 
inventory_plan = []
for product in forecast_df[product_col].unique():
    prod_df = forecast_df[forecast_df[product_col] == product]
    avg = prod_df["forecast_best"].mean() / 30
    demand = prod_df["forecast_best"].sum()
    std = prod_df["forecast_best"].std()
    eoq = np.sqrt((2 * demand * ordering_cost) / holding_cost)
    ss = z * std * np.sqrt(lead_time)
    rop = (avg * lead_time) + ss
    category = prod_df["category"].iloc[0] if "category" in forecast_df.columns else "N/A"
    inventory_plan.append({
        "Product": product,
        "Category": category,
        "AvgDailySales": round(avg, 2),
        "TotalDemand": round(demand, 2),
        "EOQ": round(eoq, 2),
        "SafetyStock": round(ss, 2),
        "ReorderPoint": round(rop, 2),
        "StdDev": round(std, 2)
    })
inv_df = pd.DataFrame(inventory_plan)

# ABC Classification
inv_df["Value"] = inv_df["TotalDemand"] * holding_cost
inv_df = inv_df.sort_values(by="Value", ascending=False)
inv_df["Cumulative%"] = inv_df["Value"].cumsum() / inv_df["Value"].sum() * 100
inv_df["ABC_Category"] = inv_df["Cumulative%"].apply(lambda x: "A" if x <= 20 else "B" if x <= 50 else "C")

# INVENTORY SIMULATION GRAPH 
row = inv_df[inv_df["Product"] == selected_product].iloc[0]
weeks = np.arange(1, 13)
avg_weekly_demand = row["AvgDailySales"] * 7

np.random.seed(int(abs(hash(selected_product)) % (2**32 - 1)))
weekly_demand = np.maximum(0, np.random.normal(avg_weekly_demand, row["StdDev"], len(weeks)))

inventory = [row["EOQ"] + row["SafetyStock"]]
for i in range(1, len(weeks)):
    next_level = inventory[-1] - weekly_demand[i]
    if next_level <= row["ReorderPoint"]:
        reorder_qty = row["EOQ"] * np.random.uniform(0.9, 1.1)
        next_level += reorder_qty
    inventory.append(next_level)

plt.figure(figsize=(6, 3.5))
plt.plot(weeks, inventory, marker="o", linewidth=2, label="Inventory Level")
plt.axhline(y=row["ReorderPoint"], color="orange", linestyle="--", label="Reorder Point")
plt.axhline(y=row["SafetyStock"], color="red", linestyle="--", label="Safety Stock")
plt.title(f"Inventory Simulation — {selected_product_display}", fontsize=11)
plt.xlabel("Weeks", fontsize=9)
plt.ylabel("Units", fontsize=9)
plt.xticks(weeks)
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close()

# METRICS
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("📦 Reorder Point", f"{row['ReorderPoint']:.2f}")
col2.metric("📈 EOQ", f"{row['EOQ']:.2f}")
col3.metric("🛡️ Safety Stock", f"{row['SafetyStock']:.2f}")

# ADDITIONAL INSIGHTS 
st.subheader("✨ Additional Insights")
st.write("#### Demand Variability & Risk")
if row["StdDev"] > row["AvgDailySales"]:
    st.warning("⚠️ High variability detected — risk of stockouts.")
else:
    st.success("✅ Demand is stable.")

st.write("#### Total Inventory Cost")
total_cost = (row["EOQ"] / 2 * holding_cost) + (ordering_cost * (row["TotalDemand"] / row["EOQ"]))
st.metric("💲 Total Inventory Cost", f"${total_cost:.2f}")

# SEASONAL INSIGHTS 
st.subheader("📆 Seasonal Insights (From Actual Sales Data)")
sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
if product_col not in sales_df.columns:
    if "product_id" in sales_df.columns:
        product_col = "product_id"
    elif "Product ID" in sales_df.columns:
        product_col = "Product ID"

prod_sales = sales_df[sales_df[product_col] == selected_product].copy()
if "units_sold" in prod_sales.columns and not prod_sales.empty:
    prod_sales["Month"] = prod_sales["date"].dt.month_name()
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    monthly = (
        prod_sales.groupby("Month")["units_sold"]
        .sum()
        .reindex(month_order, fill_value=0)
        .reset_index()
    )

    plt.figure(figsize=(8, 4))
    plt.plot(monthly["Month"], monthly["units_sold"], marker="o", linewidth=2, color="#004e92")
    plt.fill_between(monthly["Month"], monthly["units_sold"], color="#004e92", alpha=0.1)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel("Month", fontsize=9)
    plt.ylabel("Units Sold", fontsize=9)
    plt.title(f"Monthly Demand Pattern — {selected_product_display}", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    if not monthly.empty:
        peak_month = monthly.loc[monthly["units_sold"].idxmax(), "Month"]
        low_month = monthly.loc[monthly["units_sold"].idxmin(), "Month"]
        st.write(f"🌸 **Peak Month:** {peak_month}")
        st.write(f"❄️ **Lowest Month:** {low_month}")
else:
    st.warning("No historical sales data available for this product.")

# REORDER RECOMMENDATION
st.subheader("📋 Reorder Recommendation Table")
recommend = inv_df[["Product", "Category", "AvgDailySales", "ReorderPoint", "SafetyStock", "ABC_Category"]]
st.dataframe(recommend)

# DOWNLOAD BUTTON
csv = inv_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Inventory Plan (CSV)",
    data=csv,
    file_name="inventory_plan.csv",
    mime="text/csv",
    help="Click to download the full inventory optimization plan"
)

st.markdown("<hr>", unsafe_allow_html=True)
