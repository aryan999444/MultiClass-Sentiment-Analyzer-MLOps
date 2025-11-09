import streamlit as st
import sys
import os
import pandas as pd

# Adjust path to find the 'src' directory, assuming app.py is in the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the prediction function (which loads the model and vectorizer)
# NOTE: Make sure your 'predictor.py' file handles the imports correctly.
from predictor import predict_sentiment

# MLOps - Data logging path (relative to the project root)
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'feedback_log.csv')

def log_feedback(text, prediction):
    """Append user input and model prediction into csv log file."""
    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    new_data = pd.DataFrame({
        'timestamp': [pd.Timestamp.now()],
        'review_text': [text],
        'model_prediction': [prediction]
    })
    
    # Create header only if file is new
    header = not os.path.exists(LOG_FILE_PATH)
    new_data.to_csv(LOG_FILE_PATH, mode='a', index=False, header=header)

# --- Streamlit UI Setup ---

st.set_page_config(page_title="Product Sentiment Analyzer", layout="centered")

st.title("🛍️ Multi-Class Product Sentiment Analyzer")
st.markdown("---")

st.header("Analyze Your Review")

# Input text box for the user
user_input = st.text_area(
    "Enter a product review here:",
    "The camera quality is decent, but the battery life is surprisingly poor.",
    height=150
)

# Consent checkbox for MLOps data logging
consent_checkbox = st.checkbox("I consent to my review being logged for future model improvement (MLOps).")

# Button to trigger analysis
if st.button("Analyze Sentiment"):
    if not user_input:
        st.warning("Please enter some text to analyze.")
    elif not consent_checkbox:
        # User entered text but did not consent
        st.error("Please check the consent box to proceed with analysis and log data for MLOps.")
    else:
        # --- Core Logic: Analysis and MLOps Data Logging ---
        
        # 1. Get prediction from the pipeline
        sentiment = predict_sentiment(user_input)
        
        # 2. MLOps function: Log the data with prediction
        log_feedback(user_input, sentiment)
        
        # 3. Display the result with styling
        st.markdown("---") # Separator for clear output
        
        if sentiment == 'positive':
            st.success(f"**Predicted Sentiment:** {sentiment.upper()} (94% F1-Score)")
            st.markdown("This model strongly suggests a positive customer experience.")
        elif sentiment == 'negative':
            st.error(f"**Predicted Sentiment:** {sentiment.upper()} (80% F1-Score)")
            st.markdown("Attention! This indicates a high probability of a negative customer experience.")
        else: # neutral
            st.warning(f"**Predicted Sentiment:** {sentiment.upper()} (24% F1-Score)")
            st.markdown("The model detected a neutral/mixed tone. Further review is recommended.")
            
        st.caption("User data (review and prediction) has been logged to `data/feedback_log.csv` for future model retraining purposes, demonstrating the project's **MLOps capability**.")
        st.caption("Model used: Weighted Logistic Regression on TF-IDF features.")