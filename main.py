import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from itertools import combinations
from datetime import datetime
import statsmodels.api as sm
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from fbprophet import Prophet
from sklearn.metrics import r2_score, mean_squared_error

pd.options.mode.chained_assignment = None

plt.rcParams["axes.facecolor"] = "#A2A2A2"
plt.rcParams["axes.grid"] = 1

# General Infos & Playing with Features

df = pd.read_csv("Dataset\data.csv", encoding="ISO-8859-1")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df[df.Description.isnull()])
# When description is null, we have 0 unit price and missing customer ids. Let's check for whole data.

print(df[df.Description.isnull()].CustomerID.nunique())

# When description is null, we have no available customer id and zero unit price for all data. Let's drop nan values.

df = df[df.Description.notnull()]
print(df[df.CustomerID.isnull()])

# At first look, we can see records that have missing customer id, there is no specific characteristics.

# StockCode contains non-numeric records i.e. DOT. It is a cue for examining stock codes.

print("We have {} observations.".format(df.shape[0]))

df = df[df.CustomerID.notnull()]

print("We have {} observations after removing unknown customers.".format(df.shape[0]))\


print(df.isnull().sum())

# We are done with systematically missing values. But lets go deeper.

# Sometimes, missing values are filled with some denotations. "NAN", "na", "?", "Unknown", and so on. Let's check them.

print(df[df.Description.str.len() < 5])

print(df.InvoiceNo.value_counts())

# InvoiceNo has coded with 6 digit numeric characters. We can see that some InvoiceNo records starts with the letter C. This means cancellation.

print(df[df["InvoiceNo"].str.startswith("C")])

# Cancelled invoices have negative quantity.

df["Cancelled"] = df["InvoiceNo"].apply(lambda x: 1 if x.startswith("C") else 0)

print(df.head())

# Can we have both cancellation record, and record before cancellation. I mean, for example, we have C536379, have we 536379 ?

cancelled_invoiceNo = df[df.Cancelled == 1].InvoiceNo.tolist()
cancelled_invoiceNo = [x[1:] for x in cancelled_invoiceNo]

cancelled_invoiceNo[:5]

print(df[df["InvoiceNo"].isin(cancelled_invoiceNo)])

# Nothing, we have just cancellation.

# Well, maybe we have different pattern about InvoiceNo. Let's check it

print(df[df.InvoiceNo.str.len() != 6])

# No, we only have proper invoices and cancellations for InvoiceNo. We don't have any different pattern.

df = df[df.Cancelled == 0]

# Stock Codes generally contains 5 digit numerical codes.

print(df[df.StockCode.str.contains("^[a-zA-Z]")].StockCode.value_counts())

print(df[df.StockCode.str.contains("^[a-zA-Z]")].Description.value_counts())

# It looks like data contains more than customer transactions. I will drop them.

print(df[df.StockCode.str.len() > 5].StockCode.value_counts())


# Some stock codes have a letter at the end of their codes. I don't know what they refers, so I will keep them.

df =  df[~ df.StockCode.str.contains("^[a-zA-Z]")]
df["Description"] = df["Description"].str.lower()

# I  just standardize descriptions with converting them to all lowercase characters.

# Stock Codes - Description

print(df.groupby("StockCode")["Description"].nunique()[df.groupby("StockCode")["Description"].nunique() != 1])

print(df[df.StockCode == "16156L"].Description.value_counts())

print(df[df.StockCode == "17107D"].Description.value_counts())

print(df[df.StockCode == "90014C"].Description.value_counts())

# Seems we have just a litle differences between them, i.e. "," or "/"

print(df.CustomerID.value_counts())


customer_counts = df.CustomerID.value_counts().sort_values(ascending=False).head(25)

fig, ax = plt.subplots(figsize = (10, 8))

sns.barplot(y = customer_counts.index, x = customer_counts.values, orient = "h", 
            ax = ax, order = customer_counts.index, palette = "Reds_r")

plt.title("Customers that have most transactions")
plt.ylabel("Customers")
plt.xlabel("Transaction Count")
plt.savefig('Visualization\Customers that have most transactions')
# plt.show()

print(df.Country.value_counts())

country_counts = df.Country.value_counts().sort_values(ascending=False).head(25)

fig, ax = plt.subplots(figsize = (18, 10))

sns.barplot(x = country_counts.values, y = country_counts.index, orient = "h", 
            ax = ax, order = country_counts.index, palette = "Blues_r")
plt.title("Countries that have most transactions")
plt.xscale("log")
plt.savefig("Visualization\Countries that have most transactions")
# plt.show()

print(df["UnitPrice"].describe())

# 0 unit price?

print(df[df.UnitPrice == 0].head())

# I am removing them as no correlation 

print("We have {} observations.".format(df.shape[0]))

df = df[df.UnitPrice > 0]

print("We have {} observations after removing records that have 0 unit price.".format(df.shape[0]))


fig, axes = plt.subplots(1, 3, figsize = (18, 6))

sns.kdeplot(df["UnitPrice"], ax = axes[0], color = "#195190").set_title("Distribution of Unit Price")
sns.boxplot(y = df["UnitPrice"], ax = axes[1], color = "#195190").set_title("Boxplot for Unit Price")
sns.kdeplot(np.log(df["UnitPrice"]), ax = axes[2], color = "#195190").set_title("Log Unit Price Distribution")
plt.savefig('Visualization\Distribution-Boxplot-Log Unit Price Distribution')
# plt.show()

print("Lower limit for UnitPrice: " + str(np.exp(-2)))
print("Upper limit for UnitPrice: " + str(np.exp(3)))

print("quartile",np.quantile(df.UnitPrice, 0.99))

print("We have {} observations.".format(df.shape[0]))

df = df[(df.UnitPrice > 0.1) & (df.UnitPrice < 20)]

print("We have {} observations after removing unit prices smaller than 0.1 and greater than 20.".format(df.shape[0]))

fig, axes = plt.subplots(1, 3, figsize = (18, 6))

sns.kdeplot(df["UnitPrice"], ax = axes[0], color = "#195190").set_title("Distribution of Unit Price")
sns.boxplot(y = df["UnitPrice"], ax = axes[1], color = "#195190").set_title("Boxplot for Unit Price")
sns.kdeplot(np.log(df["UnitPrice"]), ax = axes[2], color = "#195190").set_title("Log Unit Price Distribution")

fig.suptitle("Distribution of Unit Price (After Removing Outliers)")
plt.savefig("Visualization\Distribution of Unit Price (After Removing Outliers)")
# plt.show()


# counter verifying for price zero 

print (df["Quantity"].describe())

# 75% 12.000000

# max 80995.000000

# Let's look at these outliers.

fig, axes = plt.subplots(1, 3, figsize = (18, 6))

sns.kdeplot(df["Quantity"], ax = axes[0], color = "#195190").set_title("Distribution of Quantity")
sns.boxplot(y = df["Quantity"], ax = axes[1], color = "#195190").set_title("Boxplot for Quantity")
sns.kdeplot(np.log(df["Quantity"]), ax = axes[2], color = "#195190").set_title("Log Quantity")
plt.savefig("Visualization\Distribution Outliers)")
# plt.show()

print("Upper limit for Quantity: " + str(np.exp(5)))

print("quartile",np.quantile(df.Quantity, 0.99))

print("We have {} observations.".format(df.shape[0]))

df = df[(df.Quantity < 150)]

print("We have {} observations after removing quantities greater than 150.".format(df.shape[0]))

fig, axes = plt.subplots(1, 3, figsize = (18, 6))

sns.kdeplot(df["Quantity"], ax = axes[0], color = "#195190").set_title("Distribution of Quantity")
sns.boxplot(y = df["Quantity"], ax = axes[1], color = "#195190").set_title("Boxplot for Quantity")
sns.kdeplot(np.log(df["Quantity"]), ax = axes[2], color = "#195190").set_title("Log Quantity")

fig.suptitle("Distribution of Quantity (After Removing Outliers)")
plt.savefig("Visualization\Distribution of Quantity (After Removing quantities greater than 150)")
plt.show()

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.drop("Cancelled", axis = 1, inplace = True)
df.to_csv("Dataset\online_retail_final.csv", index = False)
print(df.head())

# Moving towards Cohort Analysis