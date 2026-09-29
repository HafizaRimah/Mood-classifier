from sklearn.feature_extraction.text import TfidfVectorizer

# These are just two example sentences for now.
# We're using them to understand TF-IDF before applying it
# to all 16,000 training examples.
texts = [
    "I am very happy today",
    "I am very angry today"
]

# Create the TF-IDF converter.
vectorizer = TfidfVectorizer()

# Learn the vocabulary and convert the text into numbers.
X = vectorizer.fit_transform(texts)

# Show the words that TF-IDF discovered.
print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF matrix:")
print(X.toarray())