
#  Sentiment Analysis Pipeline
A comprehensive sentiment analysis system comparing Traditional Machine Learning, Deep Learning (LSTM), and Transformer (BERT) approaches on the IMDB movie reviews dataset.

##  Project Overview

This project demonstrates multiple approaches to sentiment classification:
- **Traditional ML**: TF-IDF + Logistic Regression
- **Deep Learning**: Bidirectional LSTM with embeddings
- **Transformers**: Fine-tuned BERT model

###  Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 88.5% | 88.2% | 88.9% | 88.5% |
| LSTM | 89.3% | 89.1% | 89.5% | 89.3% |
| BERT | 91.2% | 91.0% | 91.4% | 91.2% |

##  Dataset

- **Source**: IMDB Movie Reviews Dataset
- **Size**: 50,000 reviews (25,000 train, 25,000 test)
- **Classes**: Binary (Positive/Negative)
- **Balance**: Perfectly balanced dataset

##  Technologies Used

- **Python 3.8+**
- **Machine Learning**: Scikit-learn, TF-IDF Vectorization
- **Deep Learning**: TensorFlow/Keras, PyTorch
- **NLP**: NLTK, Transformers (Hugging Face)
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Deployment**: Streamlit

##  Features

### Data Analysis
-  Comprehensive EDA with visualizations
-  Word clouds for positive/negative reviews
-  Review length distribution analysis
-  Class balance verification

### Text Preprocessing
-  HTML tag removal
-  Special character cleaning
-  Stopword removal (with sentiment-bearing words retained)
-  Lemmatization
-  Tokenization

### Model Implementations

#### 1️ Traditional ML (TF-IDF + Logistic Regression)
- Fast training and inference
- Highly interpretable
- Feature importance analysis
- Production-ready

#### 2️ Deep Learning (LSTM)
- Bidirectional architecture
- Word embeddings
- Context-aware predictions
- Dropout for regularization

#### 3️ Transformer (BERT)
- Pre-trained language model
- Transfer learning
- State-of-the-art performance
- Fine-tuned on IMDB dataset

### Web Application
-  Interactive sentiment analysis
-  Real-time predictions
-  Compare multiple models
-  Confidence scores visualization

##  Model Comparison

### Performance Metrics

![Model Comparison](images/model_comparison.png)

### Key Insights

**Logistic Regression:**
-  Fastest inference (< 1ms)
-  Interpretable features
-  Smallest model size
-  Best for production at scale

**LSTM:**
-  Better context understanding
-  Handles variable-length sequences
-  Good accuracy-speed trade-off
-  Best for balanced performance

**BERT:**
-  Highest accuracy
-  Transfer learning benefits
-  Requires more resources
-  Best for maximum accuracy

##  Use Cases

-  Social media sentiment monitoring
-  Product review classification
-  Movie review aggregation
-  Customer feedback analysis
-  Brand sentiment tracking

