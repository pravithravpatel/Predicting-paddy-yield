# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 19:24:47 2025

@author: appaduraip
"""

# Set environment variable to suppress MKL memory leak warning on Windows
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '1'
np.random.seed(42)

#  Create a original Dataset 
data_size = 501
per_cluster = data_size // 3

#for 3 distinct clusters (Low, Medium, High input/yield profiles)
# input for Columns from Soil_pH, Rainfall_mm, Fertiliser_kg, Pesticide_L, Yield_kg
import numpy as np
cluster_means = np.array([
    [5.8, 800, 30, 10, 1800],  # Cluster 0: Low Input to Low Yield
    [6.5, 1200, 55, 20, 2800], # Cluster 1: Medium Input to Medium Yield
    [7.0, 1500, 80, 35, 3800]  # Cluster 2: High Input to High Yield
])
std_dev = np.array([0.2, 150, 8, 5, 300]) # Standard deviation for all

# Generate data cluster
data_list = []
for i in range(3):
    num_samples = per_cluster if i < 2 else data_size - 2 * per_cluster
    cluster_data = np.random.normal(loc=cluster_means[i], scale=std_dev, size=(num_samples, 5))
    data_list.append(cluster_data)

original_data = np.vstack(data_list)
np.random.shuffle(original_data) #shuffle


import pandas as pd
paddy = pd.DataFrame(original_data, columns=['Soil_pH', 'Rainfall_mm', 'Fertiliser_kg', 'Pesticide_L', 'Yield_kg'])
paddy.to_csv('paddydataset.csv', index=False)
print("Original 'paddydataset.csv' created.")

#data exploration and preprocessing 
print("\nData Head:")
print(paddy.head())
print("\nData Info:")
print(paddy.info())

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(paddy)

#K-Means Clustering - Optimal k determination Elbow Method
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for K-Means Clustering')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.xticks(K_range)
plt.savefig('K_Means_Elbow_Method.png')
plt.show()

# Based on the plot K=3 for this paddy data
optimal_k = 3

#Applying the Three Clustering Algorithms 

# K-Means
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(X_scaled)
kmeans_score = silhouette_score(X_scaled, kmeans_labels)

# Agglomerative Clustering
agg_model = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
agg_labels = agg_model.fit_predict(X_scaled)
agg_score = silhouette_score(X_scaled, agg_labels)

# DBSCAN
dbscan_model = DBSCAN(eps=0.8, min_samples=10)
dbscan_labels = dbscan_model.fit_predict(X_scaled)

# DBSCAN Silhouette Score
non_noise_indices = dbscan_labels != -1
if len(np.unique(dbscan_labels[non_noise_indices])) > 1:
    dbscan_score = silhouette_score(X_scaled[non_noise_indices], dbscan_labels[non_noise_indices])
else:
    dbscan_score = np.nan

#performance metrics
print("\nClustering Algorithm Performance (Silhouette Score):")
print(f"K-Means (K={optimal_k}): {kmeans_score:.4f}")
print(f"Agglomerative Clustering (K={optimal_k}): {agg_score:.4f}")
print(f"DBSCAN (eps=0.8, min_samples=10): {dbscan_score:.4f} (Noise points: {np.sum(dbscan_labels == -1)})")

#Visualization using PCA 
pca = PCA(n_components=2, random_state=42)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['KMeans_Cluster'] = kmeans_labels.astype(str)

plt.figure(figsize=(10, 7))
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x='PC1', y='PC2', hue='KMeans_Cluster', data=pca_df, palette='viridis', legend='full', s=100, alpha=0.7)
plt.title(f'K-Means Clustering of Paddy Farmers (K={optimal_k}) - PCA Reduced Data')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True)
plt.legend(title='Cluster')
plt.savefig('K_Means_PCA_Plot.png')
plt.show()

#Cluster Profiling Statistics for K-Means
paddy['KMeans_Cluster'] = kmeans_labels
cluster_profiles = paddy.groupby('KMeans_Cluster').agg({
    'Soil_pH': ['mean', 'std'],
    'Rainfall_mm': ['mean', 'std'],
    'Fertiliser_kg': ['mean', 'std'],
    'Pesticide_L': ['mean', 'std'],
    'Yield_kg': ['mean', 'std', 'count']
}).round(2)

print("\nK-Means Cluster Profiles:")
print(cluster_profiles)

# cluster profiles toCSV for the report
cluster_profiles.to_csv('KMeans_Cluster_Profiles.csv')
print("'KMeans_Cluster_Profiles.csv' saved.")

#clustering dendrogram
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
sample_X_scaled = X_scaled[:100]
linked = linkage(sample_X_scaled, method='ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Sample Data)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.savefig('Hierarchical_Dendrogram.png')
plt.show()