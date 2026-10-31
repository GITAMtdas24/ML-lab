# Experiment 8: Decision tree classifier on a binary classification task
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Binary dataset: malignant (0) vs benign (1)
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

# 2. Train the decision tree (max_depth limits overfitting)
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42).fit(X_train, y_train)
y_pred = tree.predict(X_test)

# 3. Evaluate
print(f"Train accuracy: {tree.score(X_train, y_train) * 100:.2f}%")
print(f"Test accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"F1-score : {f1_score(y_test, y_pred):.3f}\n")
print(classification_report(y_test, y_pred, target_names=data.target_names))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 4. Top feature importances
imp = sorted(zip(data.feature_names, tree.feature_importances_), key=lambda x: -x[1])[:5]
print("\nTop 5 important features:")
for name, v in imp:
    print(f"  {name}: {v:.3f}")

# 5. Visualize the tree and confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [3, 1]})
plot_tree(tree, feature_names=data.feature_names, class_names=data.target_names, filled=True, fontsize=7, ax=ax[0])
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=data.target_names, ax=ax[1])
plt.tight_layout()
plt.savefig("prog8_tree.png", dpi=110)
plt.show()