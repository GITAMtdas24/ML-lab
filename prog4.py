import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def supervised_learning_demo():
    print("=" * 60)
    print("SUPERVISED LEARNING: Classifying Iris species")
    print("=" * 60)

    # Labeled dataset: X = features, y = known labels (species)
    data = load_iris()
    X, y = data.data, data.target
    class_names = data.target_names

    # Split into train/test so we can evaluate generalization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train a classifier on labeled examples (X_train -> y_train)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Predict labels for unseen data and evaluate against ground truth
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Predict a single new, unlabeled flower measurement
    new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # sepal/petal length & width
    predicted_class = class_names[clf.predict(new_sample)[0]]
    print(f"Prediction for a new sample {new_sample.tolist()}: {predicted_class}")

    return X, y, class_names


def unsupervised_learning_demo(X, y_true, class_names):
    print("\n" + "=" * 60)
    print("UNSUPERVISED LEARNING: Clustering Iris measurements")
    print("=" * 60)

    # Unlabeled dataset: only X is used, y is NOT given to the model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ask K-Means to find 3 groups purely from feature similarity
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Internal metric: how well-separated are the clusters (no labels needed)
    sil_score = silhouette_score(X_scaled, cluster_labels)
    print(f"\nSilhouette score (cluster cohesion/separation): {sil_score:.3f}")

    # External check ONLY for our own curiosity: how well do the
    # discovered clusters line up with the true species labels?
    # (This comparison is not used during training.)
    ari = adjusted_rand_score(y_true, cluster_labels)
    print(f"Adjusted Rand Index vs. true species (evaluation only): {ari:.3f}")

    print("\nCluster centers (in standardized feature space):")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"  Cluster {i}: {np.round(center, 2)}")

    print("\nSample cluster assignments (first 10 points):")
    for i in range(10):
        print(
            f"  Point {i}: assigned to cluster {cluster_labels[i]} "
            f"(true species: {class_names[y_true[i]]})"
        )


if __name__ == "__main__":
    X, y, class_names = supervised_learning_demo()
    unsupervised_learning_demo(X, y, class_names)