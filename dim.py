import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("data.csv")  # Replace "data.csv" with your dataset path

# Step 1: Handle Missing Values in Numeric Columns
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Step 2: Handle Categorical Columns
categorical_columns = df.select_dtypes(exclude=['number']).columns
label_encoder = LabelEncoder()

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column].astype(str))  # Ensure consistent type

# Step 3: Define Features (X) and Target (y)
X = df.drop('product_wg_ton')  # Features (exclude the target column)
y = df['product_wg_ton']  # Target column

# Step 4: Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply PCA for Dimensionality Reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# Output Results
print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}")
