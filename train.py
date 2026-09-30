import mlflow
import mlflow.sklearn
import subprocess
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Safely compute file hash
try:
    with open("data/train.csv", "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
except Exception:
    file_hash = "mock_hash_v1"

# 2. Load and split dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Start MLflow tracking
mlflow.set_experiment("MLflow_DVC_Lab")

with mlflow.start_run():
    mlflow.log_param("dataset_version_hash", file_hash)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    
    # Explicitly trust the tree structure type for saving
    mlflow.sklearn.log_model(
        model, 
        "random_forest_model", 
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    
    print(f"Experiment completed successfully! Accuracy: {accuracy:.4f}")