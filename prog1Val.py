# 1. Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#classification_report Shows several evaluation metrics.Precision,Recall,F1-score,Support
import pandas as pd

# 2. Load the dataset
# The Iris dataset contains 150 samples across 3 flower species (Setosa, Versicolor, Virginica)
iris = load_iris() #print(iris.data)
X = iris.data  # Features: sepal length, sepal width, petal length, petal width X stores the input features.
y = iris.target  # Labels: 0 (Setosa), 1 (Versicolor), 2 (Virginica) Stores the output labels.

# Optional: View the structured data as a Pandas DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y #Adds another column.
print("--- First 5 Rows of the Dataset ---")
print(df.head(), "\n")

# 3. Split the data into Training and Temporary sets
# 60% Training and 40% Temporary
X_train, X_temp, y_train, y_temp = train_test_split(X, y,test_size=0.4,random_state=42,stratify=y)
# 4. Split the Temporary data into Validation and Test sets
# 20% Validation and 20% Test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,test_size=0.5,random_state=42,stratify=y_temp)
#X The features Y The labels.
#Using 42 ensures the split is reproducible: every time you run the code, the same training and testing sets are created.
#stratify=y Ensures every class has the same proportion in both sets.

# Display dataset sizes
print("--- Dataset Split ---")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# 5. Initialize and train the model
# We use Logistic Regression, a simple and highly effective algorithm for linear classification
model = LogisticRegression(max_iter=200) #Allow up to 200 optimization iterations while training.
model.fit(X_train, y_train) #This is where learning happens. The model examines

# 6.Validation
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("--- Validation Results ---")
print("Validation Accuracy: {:.2f}%".format(val_accuracy * 100))

# 7. Make predictions on the test data
y_test_pred = model.predict(X_test) #The trained model now looks at unseen flowers in X_test

# 8. Evaluate the model performance
accuracy = accuracy_score(y_test, y_test_pred) #Compares
print("--- Evaluation Results ---")
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# 9. Classification Report
print("Classification Report:")
print(classification_report(y_test,y_test_pred,target_names=iris.target_names))
#Precision: Of all samples predicted as a class, how many were actually that class?
#Recall: Of all actual samples of a class, how many did the model correctly identify?
#F1-score: The harmonic mean of precision and recall, balancing both.
#Support: The number of true samples for each class in the test set.

# 10. Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

"""| Actual | Predicted | Setosa | Versicolor | Virginica |
| ------------------ | ------ | ---------- | --------- |
| **Setosa**         | 10     | 0          | 0         |
| **Versicolor**     | 0      | 9          | 1         |
| **Virginica**      | 0      | 1          | 9         |

Class: setosa
TP = 10
FP = 0
FN = 0
TN = 20

Class: versicolor
TP = 9
FP = 1
FN = 1
TN = 19

Class: virginica
TP = 9
FP = 1
FN = 1
TN = 19

"""