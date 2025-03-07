import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load and Inspect Data ---
df = pd.read_csv("E:\Program Files\Heart Disease Prediction\heart.csv")  
print(df.head())
print(df.info())

# --- Data Preprocessing ---
X = df.drop(columns=["target"])
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Model Training & Evaluation ---
def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return acc

# --- Train and Evaluate Logistic Regression ---
log_reg = LogisticRegression()
log_reg_acc = train_and_evaluate_model(log_reg, "Logistic Regression")

# --- Train and Evaluate Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_acc = train_and_evaluate_model(rf, "Random Forest")

# --- Train and Evaluate K-Nearest Neighbors (KNN) ---
knn = KNeighborsClassifier(n_neighbors=5)
knn_acc = train_and_evaluate_model(knn, "KNN")

# --- Model Comparison ---
models = ["Logistic Regression", "Random Forest", "KNN"]
accuracies = [log_reg_acc, rf_acc, knn_acc]
plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies, palette="viridis")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Performance Comparison")
plt.show()

# --- Final Model Selection ---
best_model_index = np.argmax(accuracies)
print(f"Best Performing Model: {models[best_model_index]} with Accuracy: {accuracies[best_model_index]:.4f}")
