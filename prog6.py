import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
iris = load_iris()

x = iris.data
y = iris.target

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Different max_depth values
depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

results = []

# Train and evaluate model for each depth
for depth in depth_values:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    results.append({
        "max_depth": depth,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# Convert results into DataFrame
results_df = pd.DataFrame(results)

print("\nHyperparameter Experiment Results:")
print(results_df.to_string(index=False))


# -------------------------------
# Plot 1: Max Depth vs Accuracy
# -------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["max_depth"],
    results_df["Accuracy"],
    marker="o",
    linewidth=2
)

plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Effect of Max Depth on Model Performance")

plt.grid(True)
plt.show()


# -------------------------------------------
# Plot 2: Performance for Different Metrics
# -------------------------------------------

results_df.plot(
    x="max_depth",
    y=["Accuracy", "Precision", "Recall", "F1 Score"],
    marker="o",
    figsize=(9, 5)
)

plt.title("Performance for Different Hyperparameter Values")
plt.xlabel("Max Depth")
plt.ylabel("Score")

plt.grid(True)
plt.show()