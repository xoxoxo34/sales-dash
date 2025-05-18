import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Load dataset
df = pd.read_csv('web_logs_data1.csv')

# Data Security: Remove PII
pii_columns = ['Customer Name', 'Customer ID', 'IP Address']
df = df.drop(columns=pii_columns, errors='ignore')  # ignore if columns don't exist

# Fill missing values
df.fillna(0, inplace=True)

# Select features relevant for clustering
features = [
    'Country', 'Request Type', 'Resource Requested', 'Response Code',
    'Jobs Placed', 'Demo Requests', 'AI Assistant Requests',
    'Sales Channel', 'Salesperson', 'Retail Store', 'Revenue',
    'Product', 'Industry', 'Customer Segment', 'Hour', 'Conversion Rate'
]

# Check if all features exist
features = [feat for feat in features if feat in df.columns]

X = df[features].copy()

# Encode categorical variables if any remain as strings
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: Apply PCA for visualization & performance
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Determine optimal number of clusters using silhouette score
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

best_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters determined: {best_k}")

# Fit KMeans with optimal clusters
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = cluster_labels

# Save the model and preprocessing objects for deployment
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')

# Output silhouette score for evaluation
print(f"Silhouette Score for {best_k} clusters: {silhouette_score(X_scaled, cluster_labels)*100:.2f}%")