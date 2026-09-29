import joblib


# --------------------------------------------------
# 1. Load the saved model and vectorizer
# --------------------------------------------------

model = joblib.load("models/emotion_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")


# --------------------------------------------------
# 2. Emotion names
# --------------------------------------------------

labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]


# --------------------------------------------------
# 3. Prediction function
# --------------------------------------------------

def predict_emotion(text):
    """
    Predict the emotion of a piece of text.

    Returns:
        predicted_emotion: the emotion with the highest probability
        probabilities: probability for every emotion
    """

    # Convert the text into TF-IDF features.
    X = vectorizer.transform([text])

    # Get the predicted class.
    prediction = model.predict(X)[0]

    # Get probabilities for all six emotions.
    probabilities = model.predict_proba(X)[0]

    predicted_emotion = labels[prediction]

    return predicted_emotion, probabilities


# --------------------------------------------------
# 4. Test the function
# --------------------------------------------------

if __name__ == "__main__":
    text = "I finally achieved my goal and I am so happy!"

    emotion, probabilities = predict_emotion(text)

    print("Text:")
    print(text)

    print("\nPredicted emotion:")
    print(emotion)

    print("\nEmotion probabilities:")

    for label, probability in zip(labels, probabilities):
        print(f"{label}: {probability:.2%}")