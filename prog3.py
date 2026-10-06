import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
X, y = load_digits(return_X_y=True)
print(f"Dataset shape: {X.shape[0]} samples x {X.shape[1]} features")
print(f"Number of classes: {len(np.unique(y))}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 2. Standardize features (mean=0, std=1)
# ---------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. Baseline classifier on full-dimensional data
# ---------------------------------------------------------
model_full = LogisticRegression(max_iter=5000)
model_full.fit(X_train, y_train)
acc_full = accuracy_score(y_test, model_full.predict(X_test))

# ---------------------------------------------------------
# 4. Apply PCA (retain 95% of variance)
# ---------------------------------------------------------
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# ---------------------------------------------------------
# 5. Classifier on PCA-reduced data
# ---------------------------------------------------------
model_pca = LogisticRegression(max_iter=5000)
model_pca.fit(X_train_pca, y_train)
acc_pca = accuracy_score(y_test, model_pca.predict(X_test_pca))

# ---------------------------------------------------------
# 6. Reconstruction error (information lost by PCA)
# ---------------------------------------------------------
X_test_reconstructed = pca.inverse_transform(X_test_pca)
reconstruction_mse = np.mean((X_test - X_test_reconstructed) ** 2)

# ---------------------------------------------------------
# 7. Print comparison results
# ---------------------------------------------------------
print("===================== RESULTS =====================")
print(f"Original dimensions      : {X.shape[1]}")
print(f"Reduced dimensions (PCA) : {pca.n_components_}")
print(f"Variance retained        : {sum(pca.explained_variance_ratio_)*100:.2f}%")
print(f"Accuracy (Full features) : {acc_full:.4f}")
print(f"Accuracy (PCA features)  : {acc_pca:.4f}")
print(f"Reconstruction MSE       : {reconstruction_mse:.4f}")
print("=====================================================")

# ---------------------------------------------------------
# 8. Visualizations
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Scree plot: cumulative explained variance
pca_full_fit = PCA().fit(X_train)
cum_var = np.cumsum(pca_full_fit.explained_variance_ratio_)
axes[0].plot(range(1, len(cum_var) + 1), cum_var, marker='o', markersize=3)
axes[0].axhline(0.95, color='r', linestyle='--', label='95% variance')
axes[0].axvline(pca.n_components_, color='g', linestyle='--',
                 label=f'{pca.n_components_} components')
axes[0].set_xlabel("Number of Components")
axes[0].set_ylabel("Cumulative Explained Variance")
axes[0].set_title("Scree Plot")
axes[0].legend()
axes[0].grid(alpha=0.3)

# (b) 2D PCA projection (for visualization only)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_train)
scatter = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap='tab10', s=10, alpha=0.7)
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].set_title("2D PCA Projection of Digits Dataset")
plt.colorbar(scatter, ax=axes[1], label="Digit Class")

plt.tight_layout()
plt.savefig("pca_lab_output.png", dpi=150)
print("\nPlot saved as 'pca_lab_output.png' in the current folder.")
plt.show()