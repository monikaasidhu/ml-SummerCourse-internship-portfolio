# Machine Learning Internship Portfolio

##  Overview
This repository contains machine learning projects completed during my summer internship course as part of the **Complete Machine Learning & NLP Bootcamp** course. The projects demonstrate end-to-end ML workflows, from data preprocessing to deployment.

---

##  Author
**Monika**  
Summer Intern Course - Machine Learning & Data Science  
**Duration**: [Start Date] - [End Date]  
**Course**: Complete Machine Learning & NLP Bootcamp with MLOps Deployment

---

##  Projects

### Mini Projects

#### 1️ [Customer Segmentation Analysis](./mini-projects/01-customer-segmentation/)
**Topics**: Clustering, PCA, Feature Engineering, RFM Analysis  
**Description**: Segmented 4,338 customers into distinct groups using K-Means clustering based on purchasing behavior.  
**Key Skills**: Data preprocessing, unsupervised learning, business analytics  
**Tools**: Python, scikit-learn, pandas, matplotlib, plotly

**Results**:
- Identified 4 customer segments with 85%+ classification accuracy
- Generated actionable business recommendations
- Created interactive visualizations for stakeholder presentation

---

#### 2️ [Sentiment Analysis Pipeline](./mini-projects/02-sentiment-analysis/)
**Topics**: NLP, Machine Learning, Data Engineering
**Description**: Built an end-to-end sentiment classification pipeline that automates data generation, text preprocessing, and multi-model benchmarking on product reviews. 
**Key Skills**: Text preprocessing (Regex/NLTK), TF-IDF Feature Engineering, Scikit-Learn Pipelines, Model Evaluation (AUC-ROC)
**Tools**: Python, scikit-learn, NLTK, Pandas, TextBlob, Pickle

**Results**:

Random Forest: 90% Accuracy (Best Model)
Naive Bayes: 89.5% Accuracy
AUC Score: 0.98
Pipeline: Fully serialized for deployment (Pickle)

---

#### 3️ [Model Deployment API](./mini-projects/03-api-deployment/)
**Topics**: MLOps, API Development, Containerization  
**Description**: Created production-ready REST API for ML model inference with proper error handling and logging.  
**Key Skills**: FastAPI, Docker, API design, model serving  
**Tools**: FastAPI, Docker, uvicorn, pydantic

**Results**:
- Response time: <100ms
- Containerized application
- Comprehensive API documentation

---

#### 4️ [Time Series Forecasting](./mini-projects/04-time-series-forecast/)
**Topics**: Time Series Analysis, Multiple Models, Validation  
**Description**: Compared ARIMA, XGBoost, and LSTM for sales forecasting with rolling window validation.  
**Key Skills**: Time series preprocessing, model comparison, forecasting  
**Tools**: Prophet, statsmodels, XGBoost, TensorFlow

**Results**:
- LSTM achieved lowest RMSE
- 14-day forecast accuracy: 91%
- Interactive dashboard created

---

###  Capstone Project

#### [Intelligent Product Recommendation System](./capstone-project/)
**Topics**: End-to-End ML Pipeline, MLOps, NLP, Deployment  
**Description**: Industry-level recommendation system combining collaborative filtering and NLP for e-commerce platform. Includes sentiment analysis, review summarization, and complete MLOps pipeline.

**Key Features**:
- Hybrid recommendation engine (collaborative + content-based)
- Real-time sentiment analysis on reviews
- Automated review summarization
- A/B testing framework
- Model monitoring and retraining pipeline
- Full CI/CD integration

**Technologies**:
- **Backend**: FastAPI, PostgreSQL
- **ML/DL**: scikit-learn, TensorFlow, transformers
- **MLOps**: MLflow, DVC, Docker, GitHub Actions
- **Frontend**: Streamlit
- **Cloud**: AWS/GCP deployment ready

**Results**:
- Recommendation accuracy: 89% (top-10)
- Sentiment analysis F1-score: 0.91
- API latency: <200ms
- Successfully handles 1000+ requests/minute

---

##  Technical Stack

### Programming Languages
- Python 3.9+

### ML/DL Frameworks
- scikit-learn
- TensorFlow
- PyTorch
- XGBoost

### NLP Libraries
- transformers (Hugging Face)
- NLTK
- spaCy

### Data Processing
- pandas
- numpy
- polars

### Visualization
- matplotlib
- seaborn
- plotly
- Streamlit

### MLOps Tools
- MLflow (experiment tracking)
- DVC (data version control)
- Docker (containerization)
- FastAPI (model serving)
- GitHub Actions (CI/CD)

### Cloud & Deployment
- AWS/GCP
- Docker
- Kubernetes (basic)

---


##  Performance Metrics

| Project | Metric | Score |
|---------|--------|-------|
| Customer Segmentation | Silhouette Score | 0.375 |
| Sentiment Analysis | F1-Score | 0.90 |
| API Deployment | Response Time | <100ms |
| Time Series Forecast | RMSE | X.XX |
| Recommendation System | Accuracy@10 | 89% |

---


##  Links

- **Course**: [Complete ML & NLP Bootcamp](https://www.udemy.com/course/complete-machine-learning-nlp-bootcamp-mlops-deployment/)

---

