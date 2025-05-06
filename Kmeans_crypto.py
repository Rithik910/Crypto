import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score  # Import silhouette_score

# Load the data from the CSV string

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score  # Import silhouette_score

# Load the data from the CSV string

df = pd.read_csv("/content/coin_query_relation.csv")

# Encode 'coin' and 'query' columns
coin_encoder = LabelEncoder()
query_encoder = LabelEncoder()

df['coin_encoded'] = coin_encoder.fit_transform(df['coin'])
df['query_encoded'] = query_encoder.fit_transform(df['query'])

# Select the features for clustering
X = df[['coin_encoded', 'query_encoded']]

# Determine the optimal number of clusters (k) using the Elbow Method
inertia = []
silhouette_scores = [] # Store silhouette scores
k_range = range(2, 11)  # Test k from 2 to 10 (silhouette needs at least 2 clusters)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(X, kmeans.labels_) # Calculate silhouette score
    silhouette_scores.append(silhouette_avg)
    print(f"For k={k}, silhouette score is {silhouette_avg:.3f}")

# Plot the Elbow Method graph
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Plot Silhouette Score
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.show()

# Based on the Elbow Method graph and Silhouette Score, choose an appropriate k.  Let's assume k=3 for this example.
k = 6

# Apply k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# Print the cluster assignments
print(df[['coin', 'query', 'cluster']])

# Print cluster centers (optional, for interpretation)
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Calculate and print the overall Silhouette Score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f"\nOverall Silhouette Score: {silhouette_avg:.3f}")

# Visualize the clusters (if you have only 2 features)
if X.shape[1] == 2:
    plt.scatter(X['coin_encoded'], X['query_encoded'], c=df['cluster'], cmap='viridis')
    plt.xlabel('Coin Encoded')
    plt.ylabel('Query Encoded')
    plt.title(f'K-means Clustering (k={k})')
    plt.show()
coin_encoder = LabelEncoder()
query_encoder = LabelEncoder()

df['coin_encoded'] = coin_encoder.fit_transform(df['coin'])
df['query_encoded'] = query_encoder.fit_transform(df['query'])

# Select the features for clustering
X = df[['coin_encoded', 'query_encoded']]

# Determine the optimal number of clusters (k) using the Elbow Method
inertia = []
silhouette_scores = [] # Store silhouette scores
k_range = range(2, 11)  # Test k from 2 to 10 (silhouette needs at least 2 clusters)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(X, kmeans.labels_) # Calculate silhouette score
    silhouette_scores.append(silhouette_avg)
    print(f"For k={k}, silhouette score is {silhouette_avg:.3f}")

# Plot the Elbow Method graph
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Plot Silhouette Score
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.show()

# Based on the Elbow Method graph and Silhouette Score, choose an appropriate k.  Let's assume k=3 for this example.
k = 6

# Apply k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# Print the cluster assignments
print(df[['coin', 'query', 'cluster']])

# Print cluster centers (optional, for interpretation)
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Calculate and print the overall Silhouette Score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f"\nOverall Silhouette Score: {silhouette_avg:.3f}")

# Visualize the clusters (if you have only 2 features)
if X.shape[1] == 2:
    plt.scatter(X['coin_encoded'], X['query_encoded'], c=df['cluster'], cmap='viridis')
    plt.xlabel('Coin Encoded')
    plt.ylabel('Query Encoded')
    plt.title(f'K-means Clustering (k={k})')
    plt.show()
