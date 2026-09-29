from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import joblib
# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

dataset = load_dataset("dair-ai/emotion")

# Get the text from the training and test sets
train_texts = dataset["train"]["text"]
test_texts = dataset["test"]["text"]

# Get the corresponding emotion labels
y_train = dataset["train"]["label"]
y_test = dataset["test"]["label"]

# --------------------------------------------------
# 2. Convert text into TF-IDF features
# --------------------------------------------------

vectorizer = TfidfVectorizer()

# Learn the vocabulary and TF-IDF statistics
# ONLY from the training data.
X_train = vectorizer.fit_transform(train_texts)

# Convert the test text using the vocabulary
# learned from the training data.
X_test = vectorizer.transform(test_texts)

# --------------------------------------------------
# 3. Create the classifier
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)

# --------------------------------------------------
# 4. Train the model
# --------------------------------------------------

model.fit(X_train, y_train)
# --------------------------------------------------
# 4. Save the trained model and TF-IDF vectorizer
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/emotion_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

print("\nModel and vectorizer saved!")
# --------------------------------------------------
# 5. Make predictions
# --------------------------------------------------

predictions = model.predict(X_test)

print("Training completed!")
print(f"Training examples: {len(train_texts)}")
print(f"Test examples: {len(test_texts)}")
print(f"Predictions made: {len(predictions)}")

# --------------------------------------------------
# 6. Evaluate the model
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(accuracy)

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))