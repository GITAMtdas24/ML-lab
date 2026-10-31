# Experiment 9: K-Means clustering and visualization of the clusters
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Dataset (unlabeled for clustering purposes)
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.9, random_state=42)
X = StandardScaler().fit_transform(X)

# 2. Choose k using the Elbow method (inertia) and Silhouette score
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil[-1]:.3f}")
best_k = list(ks)[sil.index(max(sil))]
print(f"\nBest k by silhouette score: {best_k}")

# 3. Final model
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)

# 4. Visualize
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(list(ks), inertia, marker='o')
ax[0].set(xlabel="k", ylabel="Inertia", title="Elbow method")
ax[1].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='viridis', s=20)
ax[1].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', marker='X', s=200, edgecolor='k', label="Centroids")
ax[1].set(xlabel="Feature 1", ylabel="Feature 2", title=f"K-Means clusters (k={best_k})")
ax[1].legend()
plt.tight_layout()
plt.savefig("prog9_kmeans.png", dpi=120)
plt.show()