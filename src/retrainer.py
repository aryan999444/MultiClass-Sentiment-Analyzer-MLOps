import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from predictor import clean_text
from scipy.sparse import save_npz


# File paths
RAW_DATA_PATH = '../data/Dataset-SA.csv'
FEEDBACK_PATH = '../data/feedback_log.csv'
MODEL_OUTPUT_PATH = '../model/sentiment_classifier_v2.joblib'
VECT_OUTPUT_PATH = '../model/tfidf_vectorizer_v2.joblib'

def retrain_model():
    print("--- MLOps Retraining Pipeline ")
    try:
        df_original = pd.read_csv(RAW_DATA_PATH, encoding='latin-1')
        df_feedback = pd.read_csv(FEEDBACK_PATH, encoding='latin-1')
        print(f"Original data size: {len(df_original)}")
        print(f"Feedback data size: {len(df_feedback)}")

        df = df_original.rename(columns={'Review': 'review_text', 'Sentiment': 'sentiment_label'})
        df.dropna(subset=['review_text', 'sentiment_label'], inplace=True)
    except FileNotFoundError:
        print("Error: Required data files not found")
        return
    
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    X = df['cleaned_text']
    y = df['sentiment_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44, stratify=y)

    # vectorization
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )
    X_train_vectorized = vectorizer.fit_transform(X_train)

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights_dict = dict(zip(classes, weights))

    lr_classifier = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        random_state=44,
        class_weight=class_weights_dict
    )
    lr_classifier.fit(X_train_vectorized, y_train)

    # Model saving
    joblib.dump(lr_classifier, MODEL_OUTPUT_PATH)
    joblib.dump(vectorizer, VECT_OUTPUT_PATH)
    print(f"New model saved to {MODEL_OUTPUT_PATH}")
    print("MLOps Completed")

if __name__ == '__main__':
    retrain_model()