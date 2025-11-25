"""
Analysis script for predicting employee attrition.

This script reads the IBM HR Analytics Employee Attrition dataset, performs
basic cleaning and preprocessing, explores the data, trains machine‑learning
models (logistic regression, random forest and gradient boosting) using a
scikit‑learn pipeline, evaluates their performance and exports
predictions and model coefficients.  The code is written in simple,
straightforward Python with comments explaining each step.

To run this script:

    python analysis.py

Ensure that the dataset ``attrition.csv`` is located in the same
directory as this script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


def load_data(path: str) -> pd.DataFrame:
    """Load the HR attrition dataset from a CSV file."""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    """Remove constant columns and separate features and target.

    Returns X (features), y (target), and lists of categorical and numeric
    column names.
    """
    df = df.copy()
    # Drop constant columns
    constant_cols = [
        'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'
    ]
    df.drop(columns=constant_cols, inplace=True, errors='ignore')
    # Binary target
    df['AttritionFlag'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    # Separate features and target
    X = df.drop(columns=['Attrition', 'AttritionFlag'])
    y = df['AttritionFlag']
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return X, y, categorical_cols, numeric_cols


def build_pipelines(categorical_cols, numeric_cols):
    """Create preprocessing transformer and model pipelines."""
    # Preprocess categorical features: one‑hot encode
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    # Preprocess numeric features: standardise
    numeric_transformer = StandardScaler()
    # Combine into column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', categorical_transformer, categorical_cols),
            ('numeric', numeric_transformer, numeric_cols)
        ]
    )
    # Define models with balanced class weights
    logistic = LogisticRegression(max_iter=1000, class_weight='balanced')
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    # Create full pipelines
    logistic_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', logistic)
    ])
    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', rf)
    ])
    gb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', gb)
    ])
    return logistic_pipe, rf_pipe, gb_pipe


def evaluate_model(name: str, model, X_test, y_test):
    """Compute accuracy and AUC for the test set and print a summary."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Model: {name}")
    print(f"  Accuracy: {acc:.2f}")
    print(f"  ROC AUC : {auc:.2f}")
    print("  Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("  Classification Report:\n", classification_report(y_test, y_pred))
    return acc, auc


def main():
    # Load data
    df = load_data('attrition.csv')
    # Preprocess
    X, y, categorical_cols, numeric_cols = preprocess_data(df)
    # Build pipelines
    logistic_pipe, rf_pipe, gb_pipe = build_pipelines(categorical_cols, numeric_cols)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    # Train models
    logistic_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)
    gb_pipe.fit(X_train, y_train)
    # Evaluate
    results = {}
    for name, model in [('Logistic Regression', logistic_pipe),
                        ('Random Forest', rf_pipe),
                        ('Gradient Boosting', gb_pipe)]:
        acc, auc = evaluate_model(name, model, X_test, y_test)
        results[name] = {'accuracy': acc, 'auc': auc}
    # Cross‑validation for logistic regression
    cv_scores = cross_val_score(logistic_pipe, X, y, cv=5, scoring='roc_auc')
    print(f"\nLogistic Regression cross‑validation AUC scores: {cv_scores}")
    print(f"Mean CV AUC: {cv_scores.mean():.2f}")
    # Export predictions from the best model (random forest for highest accuracy)
    best_model = rf_pipe
    test_proba = best_model.predict_proba(X)[:, 1]
    predictions = pd.DataFrame({
        'EmployeeID': df.index,
        'AttritionProbability': test_proba
    })
    predictions.to_excel('predictions.xlsx', index=False)
    # Export logistic regression coefficients
    # To access coefficients, we need the names after one‑hot encoding
    ohe = logistic_pipe.named_steps['preprocessor'].named_transformers_['categorical']
    numeric_cols_transformed = numeric_cols
    categorical_feature_names = ohe.get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([categorical_feature_names, numeric_cols_transformed])
    coefs = logistic_pipe.named_steps['model'].coef_.flatten()
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    coef_df.sort_values(by='Coefficient', ascending=False, inplace=True)
    coef_df.to_csv('logistic_coefficients.csv', index=False)
    # Save evaluation summary
    with open('model_evaluation.txt', 'w') as f:
        for name, metrics in results.items():
            f.write(f"{name} - Accuracy: {metrics['accuracy']:.2f}, AUC: {metrics['auc']:.2f}\n")
        f.write(f"Mean CV AUC (Logistic): {cv_scores.mean():.2f}\n")
    print("\nOutputs saved: predictions.xlsx, logistic_coefficients.csv, model_evaluation.txt")


if __name__ == '__main__':
    main()