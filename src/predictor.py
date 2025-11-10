import joblib
import re
import os
import numpy as np
import nltk 

# --- NLTK Data Download (resilient) ---
try:
    # Using quiet=True prevents excessive output logs on Streamlit
    nltk.download('stopwords', quiet=True) 
except Exception as e:
    # This block is here for resilience, but the primary goal is success above.
    print(f"NLTK download failed (may still cause issues): {e}")
# --------------------------------------------------------

# --- 1. Paths and lazy asset loading ---

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'sentiment_classifier_weighted_lr.joblib'))
VECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'vectorizer.joblib'))

vectorizer = None
model = None

def _load_assets():
    """Load model and vectorizer into module-level variables (idempotent)."""
    global vectorizer, model
    if vectorizer is not None and model is not None:
        return

    if not os.path.exists(VECT_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model or vectorizer not found. Expected at:\n  {VECT_PATH}\n  {MODEL_PATH}")

    vectorizer = joblib.load(VECT_PATH)
    model = joblib.load(MODEL_PATH)


# --- 2. Text Cleaning Function (resilient to missing NLTK data) ---
try:
    # This import will now succeed because we forced the download above.
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    # Fallback small stopword set if NLTK data is not available in the environment
    STOPWORDS = set([
        'a', 'an', 'the', 'and', 'or', 'if', 'in', 'on', 'at', 'for', 'to', 'is', 'it', 'this', 'that',
        'of', 'with', 'as', 'by', 'from', 'was', 'were', 'be', 'been', 'are'
    ])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text.strip()


# --- 3. The Main Prediction Function ---
def predict_sentiment(raw_text):
    """Takes raw text and returns the predicted sentiment as a lowercase string.

    This function lazily loads model assets and raises informative errors when
    assets are missing.
    """
    # Ensure assets are loaded
    _load_assets()

    # 1. Clean the input text
    cleaned_text = clean_text(raw_text)

    # 2. Vectorize the cleaned text (must be a list)
    text_vectorized = vectorizer.transform([cleaned_text])

    # 3. Predict the sentiment
    prediction = model.predict(text_vectorized)

    # 4. Normalize and return the result
    # If the model returns bytes or non-str, convert to str and lower-case
    pred = prediction[0]
    try:
        return str(pred).lower()
    except Exception:
        return pred


if __name__ == '__main__':
    # Quick smoke test when running locally
    try:
        test_review = "This product is absolutely amazing and worth every penny."
        result = predict_sentiment(test_review)
        print(f"Test Review: '{test_review}' -> Prediction: {result}")
    except Exception as e:
        print("Error when testing prediction:", e)