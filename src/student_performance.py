import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
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
for col in df.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("Final_Result", axis=1)
y = df["Final_Result"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# MODEL 1: Decision Tree
# =========================================================
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:", dt_accuracy)

# =========================================================
# MODEL 2: Logistic Regression
# =========================================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

print("Logistic Regression Accuracy:", lr_accuracy)

# -----------------------------
# Save accuracy comparison
# -----------------------------
with open("outputs/model_accuracy.txt", "w") as f:
    f.write(f"Decision Tree Accuracy: {dt_accuracy}\n")
    f.write(f"Logistic Regression Accuracy: {lr_accuracy}\n")

# -----------------------------
# Classification Report
# -----------------------------
report = classification_report(y_test, y_pred_dt)
print("\nClassification Report (Decision Tree):\n", report)

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
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png", bbox_inches='tight')
plt.close()

# -----------------------------
# Feature Importance
# -----------------------------
importance = dt_model.feature_importances_

plt.figure(figsize=(10,6))
plt.barh(X.columns, importance)
plt.xlabel("Importance Score")
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
plt.bar(models, accuracies)
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
    "Actual": y_test,
    "Predicted_DT": y_pred_dt,
    "Predicted_LR": y_pred_lr
})

results["Actual_Label"] = results["Actual"].map({0: "Fail", 1: "Pass"})
results["DT_Label"] = results["Predicted_DT"].map({0: "Fail", 1: "Pass"})
results["LR_Label"] = results["Predicted_LR"].map({0: "Fail", 1: "Pass"})

results.to_csv("outputs/results.csv", index=False)

print("\nStudent Performance Prediction Completed")