import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
data = pd.read_csv("vaccine_data.csv")

data = data.dropna()
# Preprocessing: Handle categorical data via one-hot encoding
data = pd.get_dummies(data)

# Split dataset into features and target
X = data.drop("h1n1_vaccine", axis=1)
y = data["h1n1_vaccine"]

# Save the feature names
with open("feature_names.pkl", "wb") as feature_file:
    pickle.dump(X.columns.tolist(), feature_file)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model and feature names saved successfully.")
