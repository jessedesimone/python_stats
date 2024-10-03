#!/usr/bin/env python3

'''
module for k-means cluster analysis
'''

# Import packages
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Excel file
file_path = '/path/to/your/file.xlsx'  # Update with your actual file path
df = pd.read_excel(file_path)

# Extract the 'probability' column
probabilities = df['probability'].values.reshape(-1, 1)

# Perform K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(probabilities)

# View the cluster assignments
print(df.head())

# Optional: Visualize the clusters
plt.scatter(df['subject'], df['probability'], c=df['cluster'], cmap='viridis')
plt.xlabel('Subject')
plt.ylabel('Probability')
plt.title('K-means Clustering of Probabilities')
plt.show()