import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv("data.csv", low_memory=False)
current_year = datetime.now().year

if "wh_est_year" in df.columns:
    df["wh_est_year"] = pd.to_numeric(df["wh_est_year"], errors='coerce')
    df["wh_est_year"] = df["wh_est_year"].fillna(df["wh_est_year"].median() if df["wh_est_year"].notna().sum() > 0 else 2000)
    df["warehouse_age"] = current_year - df["wh_est_year"]
    df["is_warehouse_old"] = (df["warehouse_age"] > 20).astype(int)

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

clustering_columns = ['zone', 'wh_owner_type', 'govt_check_l3m']

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[clustering_columns])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[clustering_columns])

df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
df_pca['cluster'] = df['cluster']

plt.figure(figsize=(10, 6))
plt.scatter(df_pca['PCA1'], df_pca['PCA2'], c=df_pca['cluster'], cmap='viridis')
plt.title('KMeans Clustering with PCA', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=14)
plt.ylabel('PCA Component 2', fontsize=14)
plt.colorbar(label='Cluster')
plt.show()

print("Processed DataFrame with Clusters:")
print(df.head())

print(df.info())
print(df.describe())
print(df.shape)
