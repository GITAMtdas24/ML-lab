import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset (same simulated data as prog2)
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'Age': np.random.choice([25, 30, 45, 50, np.nan], size=n, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
    'Income': np.random.normal(50000, 15000, size=n),
    'Department': np.random.choice(['Tech', 'Sales', 'HR', None], size=n, p=[0.4, 0.4, 0.1, 0.1]),
    'Performance_Score': np.random.uniform(1, 5, size=n),
    'Turnover': np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
})
df['Department'] = df['Department'].where(df['Department'].notna(), np.nan)

X = df.drop(columns=['Turnover'])
y = df['Turnover']

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocessing (impute + scale numeric, impute + encode categorical)
numeric_features = ['Age', 'Income', 'Performance_Score']
categorical_features = ['Department']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ]), categorical_features),
])

# 4. Full pipeline = preprocessing + model
pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42)),
])

# 5. Hyperparameter grid
param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [3, 5, 10, None],
    'clf__min_samples_split': [2, 5],
}

# 6. Fine-tune with GridSearchCV (5-fold CV on training data only)
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print(f"Best CV F1-score: {grid.best_score_:.3f}")

# 7. Evaluate on the unseen test set
y_pred = grid.best_estimator_.predict(X_test)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=0))