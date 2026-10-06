import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def make_synthetic_dataset(n_samples=40, noise_std=0.3, random_state=42):
    """Generate synthetic data from y = sin(1.5 * pi * x) + noise."""
    rng = np.random.RandomState(random_state)
    X = np.sort(rng.uniform(0, 1, n_samples))
    true_fn = lambda x: np.sin(1.5 * np.pi * x)
    y = true_fn(X) + rng.normal(scale=noise_std, size=n_samples)
    return X.reshape(-1, 1), y, true_fn


def fit_and_evaluate(degree, X_train, y_train, X_test, y_test):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    return model, train_mse, test_mse


def main():
    X, y, true_fn = make_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Degree 1  -> too simple  -> UNDERFITTING
    # Degree 4  -> just right  -> GOOD FIT
    # Degree 15 -> too complex -> OVERFITTING
    degrees = [1, 4, 15]
    labels = ["Underfitting (degree=1)", "Good fit (degree=4)", "Overfitting (degree=15)"]

    results = []
    print(f"{'Model':<28}{'Train MSE':>12}{'Test MSE':>12}")
    print("-" * 52)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    X_plot = np.linspace(0, 1, 200).reshape(-1, 1)

    for ax, degree, label in zip(axes, degrees, labels):
        model, train_mse, test_mse = fit_and_evaluate(
            degree, X_train, y_train, X_test, y_test
        )
        results.append((label, train_mse, test_mse))
        print(f"{label:<28}{train_mse:>12.4f}{test_mse:>12.4f}")

        y_plot = model.predict(X_plot)
        ax.plot(X_plot, true_fn(X_plot), "g--", label="True function", linewidth=1.5)
        ax.plot(X_plot, y_plot, "r-", label="Model prediction", linewidth=2)
        ax.scatter(X_train, y_train, edgecolor="b", facecolor="none", label="Train data")
        ax.scatter(X_test, y_test, edgecolor="orange", facecolor="none", marker="s", label="Test data")
        ax.set_title(f"{label}\nTrain MSE={train_mse:.3f}, Test MSE={test_mse:.3f}")
        ax.set_xlabel("x")
        ax.set_ylim(-2, 2)

    axes[0].set_ylabel("y")
    axes[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig("overfitting_underfitting_demo.png", dpi=150)
    print("\nSaved plot to overfitting_underfitting_demo.png")

    print("\nInterpretation:")
    print("  - Degree 1 has HIGH error on train AND test -> underfits (too rigid).")
    print("  - Degree 4 has LOW error on both -> generalizes well.")
    print("  - Degree 15 has near-ZERO train error but HIGH test error")
    print("    -> overfits (memorized noise instead of the true pattern).")


if __name__ == "__main__":
    main()