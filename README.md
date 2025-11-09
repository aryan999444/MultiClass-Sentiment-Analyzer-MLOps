# 🚀 **Multi-Class Product Sentiment Analyzer with MLOps**

## 📝 Project Overview

This repository contains a robust, production-ready **Multi-Class Product Sentiment Analyzer** designed to tackle severe class imbalance in real-world product review data. The solution leverages advanced **MLOps** practices and a custom **class-weighted Logistic Regression** model to deliver reliable, actionable sentiment predictions for e-commerce feedback.

---

## 🌟 Key Features

- **MLOps-Driven Feedback Loop**: Real-time user feedback is logged and used to retrain the model, ensuring continuous improvement and adaptation to new data.
- **Class Imbalance Handling**: Implements **weighted Logistic Regression** to address extreme class imbalance (81% Positive, <5% Neutral), dramatically improving minority class performance.
- **Streamlit Web App**: Intuitive, interactive front-end for live predictions and feedback collection.
- **Modular Codebase**: Clean separation of data, model, and application logic for easy maintenance and extensibility.

---

## 🗂️ Repository Structure

```
├── app.py                  # Streamlit web application (front-end)
├── data/                   # Datasets and feedback logs
│   ├── Dataset-SA.csv      # Main product reviews dataset
│   └── feedback_log.csv    # User feedback for MLOps loop
├── model/                  # Trained model, vectorizer, and data splits
│   ├── sentiment_classifier_weighted_lr.joblib
│   ├── vectorizer.joblib
│   └── ...
├── notebooks/              # Jupyter notebooks for EDA and model training
│   ├── Data_Cleaning_EDA.ipynb
│   └── Model_Training_Evaluation.ipynb
├── src/                    # Core source code
│   ├── predictor.py        # Inference logic
│   └── retrainer.py        # Automated retraining logic
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/aryan999444/MultiClass-Sentiment-Analyzer-MLOps.git
   cd MultiClass-Sentiment-Analyzer-MLOps
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```powershell
   streamlit run app.py
   ```

---

## 🧠 Model & MLOps Highlights

- **Class Weighting:**
  - Neutral class weight: **~6.83**
  - Negative class weight: **~2.46**
  - Positive class weight: **1.0** (reference)
  - *This weighting scheme enables the model to learn from rare Neutral/Negative examples, overcoming the bias toward the majority class.*

- **MLOps Feedback Loop:**
  - User feedback is logged via the app and used to retrain the model automatically, closing the loop between deployment and continuous learning.

---

## 📊 Model Performance Highlights

| Metric                  | Unweighted | Weighted |
|-------------------------|------------|----------|
| Neutral F1 Score        | 0.00       | **0.24** |
| Negative F1 Score       | 0.80       | 0.80     |
| Overall Accuracy        | 89-91%     | 89-91%   |

- **Key Result:** Weighted Logistic Regression achieves a significant boost in Neutral class F1 (from 0.00 to 0.24), solving the core business problem.

---

## 📬 Feedback & Contributions

Contributions, issues, and feedback are welcome! Please open an issue or submit a pull request to help improve this project.

---

## 📄 License

This project is licensed under the MIT License.
