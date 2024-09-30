import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import sys
from pathlib import Path
# Add the 'code' directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from datasets.preprocess import load_and_preprocess_data

# Load preprocessed data
# Split the path into its components
path_components = os.path.abspath(__file__).split(os.sep)
home_path = os.sep.join(path_components[:path_components.index('PML-Assignment') + 1])
X_train, X_test, y_train, y_test = load_and_preprocess_data(os.path.join(home_path, 'data', 'titanic', 'train.csv'))
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Train a simple Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) 

# Save the model
joblib.dump(model, os.path.join(home_path, "models" , "model.pkl"))

# Print accuracy on the test set
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
