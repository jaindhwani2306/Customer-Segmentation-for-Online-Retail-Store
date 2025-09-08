import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jcopml.plot import plot_missing_value
from lifetimes.utils import summary_data_from_transaction_data
df= pd.read_csv("C:/Users/Dell/Documents/my projects/online retail/OnlineRetail.csv",encoding="unicode_escape")
print(df.head())


#missing values
print(df.isnull().sum().sum())
print(plot_missing_value(df, return_df = True))

df.describe()
#outliers for quantity column
Q1 = df['Quantity'].quantile(0.25)
Q3 = df["Quantity"].quantile(0.75)
IQR= Q3 - Q1
lb = float(Q1) - (1.5*IQR)
ub = float(Q3) + (1.5*IQR)
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("LOWER BOUND:",lb)
print("UPPER BOUND:",ub)

#outlier for unit price
Q1_unit=df['UnitPrice'].quantile(0.25)
Q3_unit=df['UnitPrice'].quantile(0.75)
IQR_unit=Q3_unit- Q1_unit
lb_unit=float(Q1_unit)-(1.5*IQR_unit)
ub_unit=float(Q3_unit)+(1.5*IQR_unit)
print("Q1_UNIT:",Q1_unit)
print("Q3_UNIT:",Q3_unit)
print("IQR_UNIT:",IQR_unit)
print("lower bound_unit:",lb_unit)
print("upper bound_unit",ub_unit)
#cleaning data
# Drop missing IDs and invalid values
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Add Revenue column
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Group properly (must keep CustomerID)
orders = df.groupby(['CustomerID','InvoiceNo','InvoiceDate']).agg({'Revenue':'sum'}).reset_index()
print("Orders grouped successfully")

# RFM summary
from lifetimes.utils import summary_data_from_transaction_data
rfm = summary_data_from_transaction_data(
    orders,
    customer_id_col='CustomerID',
    datetime_col='InvoiceDate',
    monetary_value_col='Revenue'
).reset_index()
print("RFM summary created")


# Build RFM table
rfm = summary_data_from_transaction_data(
    orders, 
    customer_id_col='CustomerID', 
    datetime_col='InvoiceDate', 
    monetary_value_col='Revenue'
).reset_index()

# RFM scoring
def recency_score(data):
    if data <= 60: return 1
    elif data <= 128: return 2
    elif data <= 221: return 3
    else: return 4

def frequency_score(data):
    if data <= 1: return 1
    elif data <= 2: return 2
    elif data <= 4: return 3
    else: return 4

def monetary_value_score(data):
    if data <= 142.9: return 1
    elif data <= 292.5: return 2
    elif data <= 412.4: return 3
    else: return 4


rfm['R'] = rfm['recency'].apply(recency_score)
rfm['F'] = rfm['frequency'].apply(frequency_score)
rfm['M'] = rfm['monetary_value'].apply(monetary_value_score)
rfm['RFM_score'] = rfm[['R','F','M']].sum(axis=1)

# Segment labels
rfm['label'] = 'Bronze'
rfm.loc[rfm['RFM_score'] > 4, 'label'] = 'Silver'
rfm.loc[rfm['RFM_score'] > 6, 'label'] = 'Gold'
rfm.loc[rfm['RFM_score'] > 8, 'label'] = 'Platinum'
rfm.loc[rfm['RFM_score'] > 10, 'label'] = 'Diamond'

# Plot bar chart of segments
barplot = dict(rfm['label'].value_counts())
plt.bar(barplot.keys(), barplot.values())
plt.title("Customer Segments")
plt.show()

# --- Clustering with KMeans ---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Scale RFM metrics
X = rfm[['recency','frequency','monetary_value']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
score = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score.append(kmeans.inertia_)

plt.plot(range(1, 15), score, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()


kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)



cluster_summary = rfm.groupby('Cluster').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary_value': 'mean'
}).round(2)


def label_cluster(row):
    if row['recency'] <= cluster_summary['recency'].mean() and row['frequency'] >= cluster_summary['frequency'].mean() and row['monetary_value'] >= cluster_summary['monetary_value'].mean():
        return "VIP"
    elif row['recency'] <= cluster_summary['recency'].mean() and row['frequency'] >= cluster_summary['frequency'].mean():
        return "Loyal"
    elif row['recency'] > cluster_summary['recency'].mean() and row['frequency'] <= cluster_summary['frequency'].mean():
        return "At Risk"
    else:
        return "Regular"


rfm['Cluster_Label'] = rfm.apply(label_cluster, axis=1)


print(rfm['Cluster_Label'].value_counts())

#Plot distribution
plt.figure(figsize=(8,6))
sns.countplot(data=rfm, x='Cluster_Label', order=rfm['Cluster_Label'].value_counts().index, palette="viridis")
plt.title("Customer Segments by Business Labels")
plt.show()
