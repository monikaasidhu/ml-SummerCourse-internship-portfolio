# Mini Project 2: Sentiment Analysis Pipeline

##  Overview
Complete sentiment analysis system comparing multiple ML algorithms for text classification.

##  Objectives
- Build text preprocessing pipeline
- Compare multiple ML models
- Deploy production-ready sentiment classifier

##  Dataset
- **Source**: Generated product reviews
- **Size**: 3,000 reviews
- **Distribution**: 50% Positive, 50% Negative
- **Features**: Review text, sentiment label

##  Technologies
- **NLP**: NLTK, TextBlob
- **ML**: scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Feature Extraction**: TF-IDF, Count Vectorizer

##  Methodology

### 1. Text Preprocessing
- Lowercase conversion
- URL/email removal
- Punctuation removal
- Stopword removal
- Lemmatization

### 2. Feature Extraction
- TF-IDF Vectorization (5000 features, bigrams)
- Count Vectorization (comparison)

### 3. Models Trained
- Logistic Regression
- Naive Bayes
- Linear SVM
- Random Forest

##  Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.89 | 0.85 | 0.94 | 0.89 |
| Naive Bayes | 0.89 | 0.86 | 0.94 | 0.90 |
| Linear SVM | 0.89 | 0.86 | 0.93 | 0.89 |
| Random Forest | 0.89 | 0.86 | 0.94 | 0.90 |

**Best Model**: Random Forest (F1: 0.9019)

##  Key Findings
- TF-IDF with bigrams most effective
- Random Forest best overall predictor, accuracy, and strong F1 performance.
- Preprocessing critical for performance
- ~90% accuracy achievable
