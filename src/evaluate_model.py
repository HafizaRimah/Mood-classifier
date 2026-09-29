from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

# Create a folder for our evaluation plots.
os.makedirs("reports", exist_ok=True)

dataset = load_dataset("dair-ai/emotion")

train_texts = dataset["train"]["text"]
test_texts = dataset["test"]["text"]

y_train = dataset["train"]["label"]
y_test = dataset["test"]["label"]


# --------------------------------------------------
# 2. Convert text into TF-IDF features
# --------------------------------------------------

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)


# --------------------------------------------------
# 3. Train the model
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# --------------------------------------------------
# 4. Make predictions
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# 5. Get evaluation results
# --------------------------------------------------

labels = dataset["train"].features["label"].names

report = classification_report(
    y_test,
    predictions,
    target_names=labels,
    output_dict=True
)

confusion = confusion_matrix(y_test, predictions)


# --------------------------------------------------
# 6. Plot F1 score for each emotion
# --------------------------------------------------

f1_scores = [report[label]["f1-score"] for label in labels]

plt.figure(figsize=(8, 5))

plt.bar(labels, f1_scores)

plt.title("F1 Score by Emotion")
plt.xlabel("Emotion")
plt.ylabel("F1 Score")

plt.ylim(0, 1)

plt.tight_layout()

# Save the plot as an image.
plt.savefig("reports/f1_scores.png", dpi=300)

plt.show()
plt.close()


# --------------------------------------------------
# 7. Plot the confusion matrix
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(confusion)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)

# Write the number inside each cell
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, confusion[i, j],
                 ha="center",
                 va="center")

plt.tight_layout()

# Save the confusion matrix as an image.
plt.savefig("reports/confusion_matrix.png", dpi=300)

plt.show()
plt.close()