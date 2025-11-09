import joblib
import re
from nltk.corpus import stopwords
import numpy as np
import os

# --- 1. Load the Model Assets ---
# Define paths relative to the predictor.py script
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'sentiment_classifier_weighted_lr.joblib')
VECT_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'vectorizer.joblib')

try:
    # Load the fitted TF-IDF Vectorizer and the Weighted Logistic Regression model
    vectorizer = joblib.load(VECT_PATH)
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("Error: Model or Vectorizer files not found. Ensure they are saved in the 'model/' folder.")
    raise

# --- 2. Text Cleaning Function (from your notebook) ---
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower() 
    text = re.sub(r'[^a-z\s]', ' ', text) 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    text = text.strip()
    return text

# --- 3. The Main Prediction Function ---
def predict_sentiment(raw_text):
    """Takes raw text and returns the predicted sentiment."""
    # 1. Clean the input text
    cleaned_text = clean_text(raw_text)
    
    # 2. Vectorize the cleaned text (must be a list)
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # 3. Predict the sentiment
    prediction = model.predict(text_vectorized)
    
    # 4. Return the result (e.g., 'positive', 'negative', 'neutral')
    return prediction[0]

if __name__ == '__main__':
    # Test the function (optional)
    test_review = "This product is absolutely amazing and worth every penny."
    result = predict_sentiment(test_review)
    print(f"Test Review: '{test_review}' -> Prediction: {result}")