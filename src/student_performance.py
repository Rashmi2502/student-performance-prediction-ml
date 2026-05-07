import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# -----------------------------
# Create folders
# -----------------------------
os.makedirs("graphs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/student_data.csv")

print("Dataset Loaded Successfully")
print("Total Students:", len(df))

# -----------------------------
# Drop unnecessary column
# -----------------------------
if "Student_ID" in df.columns:
    df.drop("Student_ID", axis=1, inplace=True)

# -----------------------------
# Encode categorical variables
# -----------------------------
categorical_cols = df.select_dtypes(include=['object', 'string']).columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("Final_Result", axis=1)
y = df["Final_Result"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Scaling (for Logistic Regression)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# MODEL 1: Decision Tree
# =========================================================
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:", round(dt_accuracy, 3))

# =========================================================
# MODEL 2: Logistic Regression
# =========================================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

print("Logistic Regression Accuracy:", round(lr_accuracy, 3))

# -----------------------------
# Save accuracy comparison
# -----------------------------
with open("outputs/model_accuracy.txt", "w") as f:
    f.write(f"Decision Tree Accuracy: {dt_accuracy:.3f}\n")
    f.write(f"Logistic Regression Accuracy: {lr_accuracy:.3f}\n")

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report (Decision Tree):\n")
print(classification_report(y_test, y_pred_dt))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png", bbox_inches='tight')
plt.close()

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png", bbox_inches='tight')
plt.close()

# -----------------------------
# Feature Importance (FIXED)
# -----------------------------
importance = dt_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Remove zero-importance features
importance_df = importance_df[importance_df["Importance"] > 0]

# Sort for better visualization
importance_df = importance_df.sort_values(by="Importance", ascending=True)

plt.figure(figsize=(8,5))
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("graphs/feature_importance.png", bbox_inches='tight')
plt.close()

# -----------------------------
# Model Comparison Graph
# -----------------------------
models = ['Decision Tree', 'Logistic Regression']
accuracies = [dt_accuracy, lr_accuracy]

plt.figure(figsize=(6,4))
sns.barplot(x=models, y=accuracies)

plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.ylim(0,1)

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.savefig("graphs/model_comparison.png", bbox_inches='tight')
plt.close()

# -----------------------------
# Save Predictions
# -----------------------------
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted_DT": y_pred_dt,
    "Predicted_LR": y_pred_lr
})

results["Actual_Label"] = results["Actual"].map({0: "Fail", 1: "Pass"})
results["DT_Label"] = results["Predicted_DT"].map({0: "Fail", 1: "Pass"})
results["LR_Label"] = results["Predicted_LR"].map({0: "Fail", 1: "Pass"})

results.to_csv("outputs/results.csv", index=False)

print("\nStudent Performance Prediction Completed")