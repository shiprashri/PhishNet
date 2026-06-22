import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# STEP 1: LOAD DATA
print("=" * 50)
print("STEP 1: Loading Dataset")
print("=" * 50)

df = pd.read_csv('dataset/dataset_phishing.csv')
print(f"Total rows    : {df.shape[0]}")
print(f"Total features: {df.shape[1] - 2}")
print(f"\nLabel distribution:")
print(df['status'].value_counts())

# STEP 2: PREPARE FEATURES
print("\n" + "=" * 50)
print("STEP 2: Preparing Features")
print("=" * 50)

X = df.drop(columns=['url', 'status'])
y = df['status'].map({'phishing': 1, 'legitimate': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Features used    : {X_train.shape[1]}")

# STEP 3: TRAIN MODELS
print("\n" + "=" * 50)
print("STEP 3: Training Models")
print("=" * 50)

models = {
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = {"model": model, "accuracy": acc, "preds": preds}
    print(f"Done — Accuracy: {acc * 100:.2f}%")

# STEP 4: DETAILED REPORT
print("\n" + "=" * 50)
print("STEP 4: Detailed Results")
print("=" * 50)

for name, result in results.items():
    print(f"\n{'-' * 40}")
    print(f"Model: {name}")
    print(f"Accuracy: {result['accuracy'] * 100:.2f}%")
    print(classification_report(
        y_test, result['preds'],
        target_names=['legitimate', 'phishing']
    ))

# STEP 5: SAVE BEST MODEL
print("\n" + "=" * 50)
print("STEP 5: Saving Best Model")
print("=" * 50)

best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
print(f"Best model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)")

joblib.dump(best_model, 'model/phishing_model.pkl')
joblib.dump(X.columns.tolist(), 'model/feature_cols.pkl')
print("Saved: model/phishing_model.pkl")
print("Saved: model/feature_cols.pkl")

# STEP 6: CONFUSION MATRIX
print("\n" + "=" * 50)
print("STEP 6: Confusion Matrix (Best Model)")
print("=" * 50)

cm = confusion_matrix(y_test, results[best_name]['preds'])
print(f"""
                  Predicted
                  Legit    Phishing
Actual  Legit  [  {cm[0][0]}     {cm[0][1]}  ]
        Phish  [  {cm[1][0]}      {cm[1][1]}  ]
""")
print(f"Correctly identified {cm[0][0]} legitimate URLs")
print(f"Correctly caught     {cm[1][1]} phishing URLs")
print(f"Missed phishing      {cm[1][0]} URLs (false negatives)")
print(f"False alarms         {cm[0][1]} URLs (false positives)")

# STEP 7: FEATURE IMPORTANCE CHART
print("\n" + "=" * 50)
print("STEP 7: Feature Importance Chart")
print("=" * 50)

rf = results["Random Forest"]["model"]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]
top_features = [X.columns[i] for i in indices]
top_scores = importances[indices]

plt.figure(figsize=(12, 6))
plt.bar(range(15), top_scores, color='steelblue', edgecolor='black', linewidth=0.5)
plt.xticks(range(15), top_features, rotation=45, ha='right', fontsize=10)
plt.title('Top 15 Most Important Features — Random Forest', fontsize=14, fontweight='bold')
plt.ylabel('Importance Score')
plt.xlabel('Feature')
plt.tight_layout()
plt.savefig('static/images/feature_importance.png', dpi=150)
print("Saved: static/images/feature_importance.png")

# STEP 8: ACCURACY COMPARISON CHART
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] * 100 for m in model_names]

plt.figure(figsize=(8, 5))
colors = ['steelblue', 'coral', 'seagreen']
bars = plt.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=0.5)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1.5,
             f'{acc:.2f}%', ha='center', va='top', fontsize=12,
             fontweight='bold', color='white')
plt.ylim(85, 100)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig('static/images/model_comparison.png', dpi=150)
print("Saved: static/images/model_comparison.png")

print("\n" + "=" * 50)
print("ALL DONE — Summary:")
print("=" * 50)
for name, result in results.items():
    print(f"  {name:25s}: {result['accuracy']*100:.2f}%")
print(f"\nBest model: {best_name}")