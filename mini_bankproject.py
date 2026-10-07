# ============================================
# BANK CUSTOMER ANALYSIS PROJECT
# ============================================

# Libraries import karna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================
# STEP 1: CHECK CSV FILE
# ============================================

# Check kar rahe hain ki CSV file same folder mein hai ya nahi
if not os.path.exists("bank_customers.csv"):
    print("Error: bank_customers.csv file nahi mili.")
    print("CSV file ko bankproject.py ke same folder mein rakho.")
    exit()

print("CSV file mil gayi!")


# ============================================
# STEP 2: READ CSV FILE
# ============================================

# CSV file ko Pandas DataFrame mein load karna
df = pd.read_csv("bank_customers.csv")

print("\nData successfully loaded!")


# ============================================
# STEP 3: DISPLAY DATA
# ============================================

print("\nFirst 5 Customers:")
print(df.head())


# ============================================
# STEP 4: BASIC INFORMATION
# ============================================

# Dataset mein kitni rows aur columns hain
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# Data types
print("\nData Types:")
print(df.dtypes)


# ============================================
# STEP 5: CHECK MISSING VALUES
# ============================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================
# STEP 6: REMOVE DUPLICATE RECORDS
# ============================================

df = df.drop_duplicates()

print("\nDuplicate records removed.")


# ============================================
# STEP 7: DATA ANALYSIS
# ============================================

print("\n========== DATA ANALYSIS ==========")

# Average age
print("\nAverage Age:")
print(df["Age"].mean())

# Average income
print("\nAverage Income:")
print(df["Income"].mean())

# Average balance
print("\nAverage Balance:")
print(df["Balance"].mean())


# ============================================
# STEP 8: MEDIAN
# ============================================

print("\n========== MEDIAN ==========")

print("Median Age:", df["Age"].median())
print("Median Income:", df["Income"].median())
print("Median Balance:", df["Balance"].median())


# ============================================
# STEP 9: NUMPY ANALYSIS
# ============================================

# Pandas columns ko NumPy arrays mein convert karna
age = np.array(df["Age"])
income = np.array(df["Income"])
balance = np.array(df["Balance"])

print("\n========== NUMPY ANALYSIS ==========")

print("NumPy Average Age:", np.mean(age))
print("NumPy Average Income:", np.mean(income))
print("NumPy Average Balance:", np.mean(balance))


# ============================================
# STEP 10: ACCOUNT TYPE ANALYSIS
# ============================================

print("\n========== ACCOUNT TYPE ANALYSIS ==========")

# Savings aur Current account ka average balance
average_balance = df.groupby("Account_Type")["Balance"].mean()

print("\nAverage Balance by Account Type:")
print(average_balance)


# Savings aur Current account ka average income
average_income = df.groupby("Account_Type")["Income"].mean()

print("\nAverage Income by Account Type:")
print(average_income)


# ============================================
# STEP 11: GENDER ANALYSIS
# ============================================

print("\n========== GENDER ANALYSIS ==========")

gender_analysis = df.groupby("Gender")[["Income", "Balance"]].mean()

print(gender_analysis)


# ============================================
# STEP 12: TOP CUSTOMERS
# ============================================

print("\n========== TOP CUSTOMERS ==========")

# Highest balance wale customers
top_customers = df.sort_values(
    "Balance",
    ascending=False
).head(5)

print(top_customers)


# ============================================
# STEP 13: FILTER CUSTOMERS
# ============================================

print("\n========== HIGH INCOME CUSTOMERS ==========")

# Jinki income 500000 se zyada hai
high_income = df[df["Income"] > 500000]

print(high_income)


# ============================================
# STEP 14: CORRELATION
# ============================================

print("\n========== CORRELATION ==========")

correlation = df[
    ["Age", "Income", "Balance", "Credit_Score", "Tenure"]
].corr()

print(correlation)


# ============================================
# STEP 15: LINE CHART
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    df["Age"],
    df["Balance"],
    marker="o"
)

plt.title("Age vs Balance")
plt.xlabel("Age")
plt.ylabel("Balance")

plt.grid()

plt.show()


# ============================================
# STEP 16: BAR CHART
# ============================================

plt.figure(figsize=(7, 5))

average_balance.plot(kind="bar")

plt.title("Average Balance by Account Type")
plt.xlabel("Account Type")
plt.ylabel("Average Balance")

plt.show()


# ============================================
# STEP 17: HISTOGRAM
# ============================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Age"],
    bins=10
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.show()


# ============================================
# STEP 18: PIE CHART
# ============================================

plt.figure(figsize=(7, 7))

df["Account_Type"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Account Type Distribution")
plt.ylabel("")

plt.show()


# ============================================
# STEP 19: SCATTER PLOT
# ============================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Income"],
    df["Balance"]
)

plt.title("Income vs Balance")
plt.xlabel("Income")
plt.ylabel("Balance")

plt.show()


# ============================================
# STEP 20: BOX PLOT
# ============================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Account_Type",
    y="Balance",
    data=df
)

plt.title("Balance by Account Type")

plt.show()


# ============================================
# STEP 21: HEATMAP
# ============================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True
)

plt.title("Correlation Between Numerical Variables")

plt.show()


# ============================================
# STEP 22: SAVE CLEANED DATA
# ============================================

# Cleaned data ko new CSV file mein save karna
df.to_csv(
    "cleaned_bank_customers.csv",
    index=False
)

print("\n================================")
print("BANK CUSTOMER ANALYSIS COMPLETE!")
print("================================")

print("\nCleaned file created:")
print("cleaned_bank_customers.csv")