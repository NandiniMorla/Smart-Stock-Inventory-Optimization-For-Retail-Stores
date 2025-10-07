import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")


# Load the dataset

df = pd.read_csv("retail_sales_dataset_2022_2024.csv")
print("Data loaded successfully.")
print("Shape:", df.shape)
print(df.head())

df.columns = df.columns.str.lower()

# Basic overview
print("\nDataset Info:")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe(include='all'))


# Handle missing values

num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(exclude=['number']).columns

for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values handled successfully.")

# Convert date column

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')

# Outlier detection
if 'units_sold' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df['units_sold'])
    plt.title("Outlier Detection - Units Sold")
    plt.xlabel("Units Sold")
    plt.show()

# Sales Trend
prod_col = 'product_id' if 'product_id' in df.columns else 'product'
sales_col = 'units_sold' if 'units_sold' in df.columns else 'sales'

product_list = ['P001', 'P002', 'P003']

for pid in product_list:
    if pid in df[prod_col].unique():
        product_df = df[df[prod_col] == pid]
        plt.figure(figsize=(10, 4))
        plt.plot(product_df['date'], product_df[sales_col], marker='o', label=f"Product {pid}")
        plt.title(f"Sales Trend for {pid}")
        plt.xlabel("Date")
        plt.ylabel("Units Sold")
        plt.legend()
        plt.show()
    else:
        print(f"⚠️ Product {pid} not found in dataset")

# Monthly Sales Trend (Overall)

if 'date' in df.columns and sales_col in df.columns:
    df['month'] = df['date'].dt.to_period('M').astype(str)
    monthly_sales = df.groupby('month')[sales_col].sum().reset_index()
    plt.figure(figsize=(10, 4))
    plt.plot(monthly_sales['month'], monthly_sales[sales_col], marker='o', color='blue')
    plt.title("Monthly Sales Trend (Overall)")
    plt.xlabel("Month")
    plt.ylabel("Total Units Sold")
    plt.xticks(rotation=45)
    plt.show()

# Monthly Sales Trend by Category
if 'category' in df.columns:
    monthly_category = df.groupby(['month', 'category'])[sales_col].sum().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_category, x='month', y=sales_col, hue='category', marker='o')
    plt.title("Monthly Sales Trend by Category")
    plt.xlabel("Month")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=45)
    plt.legend(title="Category")
    plt.show()

# Total Sales by Category
if 'category' in df.columns:
    category_sales = df.groupby('category')[sales_col].sum().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=category_sales, x='category', y=sales_col, palette='viridis')
    plt.title("Total Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Units Sold")
    plt.xticks(rotation=45)
    plt.show()

# Feature Engineering
if 'date' in df.columns:
    df['month_num'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()

if 'promotion' in df.columns:
    df['promotion_flag'] = df['promotion'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

print("\nFeature engineering completed.")

# Save cleaned dataset

os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_retail_sales.csv", index=False)
print("\nCleaned dataset saved as data/cleaned_retail_sales.csv")
print("\nMilestone 1 completed successfully.")
