# train_model.py
import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# Load the dataset
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate and save the model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model trained with an accuracy of: {accuracy:.2f}")

# Save the model to a file
with open("wine_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as 'wine_model.pkl'")
