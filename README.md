# 🎭 Mood Classifier

A machine learning application that predicts emotions from text. It uses **TF-IDF** for text vectorization and **Logistic Regression** for classification, and provides an interactive web interface built with **Streamlit**.

## Features

- Text-based emotion classification.
- Six emotion classes: **sadness**, **joy**, **love**, **anger**, **fear**, and **surprise**.
- TF-IDF text vectorization.
- Logistic Regression classification.
- Prediction probabilities for all emotion classes.
- Interactive Streamlit web interface.

## Machine Learning Architecture

The application follows this pipeline:

```
User Input
    ↓
TF-IDF Vectorizer
    ↓
Logistic Regression
    ↓
Predicted Emotion
    ↓
Emotion Probabilities
```

1. The user enters a sentence.
2. The sentence is transformed into numerical features using a **TF-IDF vectorizer** trained on the dataset.
3. The features are passed to a **Logistic Regression** model.
4. The model predicts the most likely emotion.
5. The application displays **prediction probabilities** for all six emotion classes.

## Project Structure

```
Mood-classifier/
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── .gitignore
├── models/
│   ├── emotion_model.joblib        # Trained Logistic Regression model
│   └── tfidf_vectorizer.joblib     # Trained TF-IDF vectorizer
├── reports/
│   ├── confusion_matrix.png        # Confusion matrix visualization
│   └── f1_scores.png               # F1 score visualization
└── src/
    ├── download_data.py            # Dataset download and exploration
    ├── explore_data.py             # Dataset split overview
    ├── tfidf_test.py               # TF-IDF learning experiment
    ├── train.py                    # Model training script
    ├── evaluate_model.py           # Model evaluation and report generation
    └── predict.py                  # Prediction function
```

## Technologies Used

- **Python 3.11**
- **scikit-learn** — TF-IDF vectorization and Logistic Regression
- **pandas** — Data manipulation
- **joblib** — Model serialization (saving and loading)
- **datasets** — Hugging Face dataset loading ([dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion))
- **matplotlib** — Evaluation visualizations
- **Streamlit** — Interactive web interface

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/HafizaRimah/Mood-classifier.git
   cd Mood-classifier
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment (Windows):**

   ```bash
   .venv\Scripts\activate
   ```

4. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## Usage

1. Enter a sentence in the text area (e.g., *"I finally achieved my goal and I am so happy!"*).
2. Click **Analyze Mood**.
3. View the **predicted emotion** and the **probability breakdown** for all six emotion classes.

## Model Training

The trained model and vectorizer are included in the `models/` directory. If you want to retrain the model from scratch:

```bash
python src/train.py
```

This will:
- Download the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.
- Train a Logistic Regression model on TF-IDF features.
- Save the model and vectorizer to the `models/` directory.

## Limitations

- The model's predictions are **not guaranteed** to reflect a person's actual emotions.
- Predictions depend on the patterns learned from the training data.
- The probabilities are **model estimates**, not ground truth.
- The model works best with English text similar to the training data.

## Future Improvements

- Add text preprocessing (lowercasing, removing punctuation, stop-word removal).
- Train on a larger and more diverse dataset.
- Evaluate additional classification models (e.g., Support Vector Machines, Random Forests).
- Add support for multiple languages.
