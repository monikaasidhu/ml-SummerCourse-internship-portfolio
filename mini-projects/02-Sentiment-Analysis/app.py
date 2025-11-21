
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4ecdc4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 2px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎭 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Analyze the sentiment of movie reviews using Machine Learning and Deep Learning models")

# Load models
@st.cache_resource
def load_models():
    # Load Logistic Regression
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('models/text_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load LSTM
    lstm_model = load_model('models/lstm_sentiment_model.h5')
    
    with open('models/lstm_tokenizer.pkl', 'rb') as f:
        lstm_tokenizer = pickle.load(f)
    
    return lr_model, vectorizer, preprocessor, lstm_model, lstm_tokenizer

try:
    lr_model, vectorizer, preprocessor, lstm_model, lstm_tokenizer = load_models()
    models_loaded = True
except:
    st.error("⚠️ Models not found. Please run the training notebook first!")
    models_loaded = False

# Sidebar
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression (TF-IDF)", "LSTM Neural Network", "Both Models"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")

if model_choice == "Logistic Regression (TF-IDF)":
    st.sidebar.info("""
    **Traditional ML Approach**
    - TF-IDF Vectorization
    - Logistic Regression Classifier
    - Fast inference
    - Interpretable features
    """)
elif model_choice == "LSTM Neural Network":
    st.sidebar.info("""
    **Deep Learning Approach**
    - Bidirectional LSTM
    - Word Embeddings
    - Context-aware
    - Better for complex patterns
    """)
else:
    st.sidebar.info("""
    **Compare Both Models**
    - See predictions from both approaches
    - Compare confidence scores
    - Understand model differences
    """)

# Main content
if models_loaded:
    # Input section
    st.markdown("---")
    st.markdown("## 📝 Enter Your Review")
    
    # Example reviews
    examples = {
        "Positive Example": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        "Negative Example": "Terrible waste of time. Poor acting, weak storyline, and terrible special effects.",
        "Neutral Example": "It was okay, nothing special but not terrible either. Average movie."
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📗 Load Positive Example"):
            st.session_state.review_text = examples["Positive Example"]
    with col2:
        if st.button("📕 Load Negative Example"):
            st.session_state.review_text = examples["Negative Example"]
    with col3:
        if st.button("📘 Load Neutral Example"):
            st.session_state.review_text = examples["Neutral Example"]
    
    # Text input
    review_text = st.text_area(
        "Type or paste a movie review:",
        value=st.session_state.get('review_text', ''),
        height=150,
        placeholder="e.g., This movie was amazing! I loved every minute of it..."
    )
    
    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if review_text.strip() == "":
            st.warning("⚠️ Please enter a review first!")
        else:
            with st.spinner("Analyzing..."):
                # Preprocess
                cleaned_text = preprocessor.preprocess(review_text)
                
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                if model_choice in ["Logistic Regression (TF-IDF)", "Both Models"]:
                    # Logistic Regression prediction
                    features = vectorizer.transform([cleaned_text])
                    pred_lr = lr_model.predict(features)[0]
                    prob_lr = lr_model.predict_proba(features)[0]
                    
                    sentiment_lr = "Positive 😊" if pred_lr == 1 else "Negative 😞"
                    confidence_lr = prob_lr[pred_lr] * 100
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("### 🤖 Logistic Regression Model")
                        
                        if pred_lr == 1:
                            st.markdown(f'<div class="sentiment-positive"><h3>Sentiment: {sentiment_lr}</h3></div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="sentiment-negative"><h3>Sentiment: {sentiment_lr}</h3></div>', 
                                      unsafe_allow_html=True)
                        
                        st.progress(confidence_lr / 100)
                        st.markdown(f"**Confidence:** {confidence_lr:.2f}%")
                    
                    with col2:
                        st.markdown("### 📈 Probability")
                        st.metric("Positive", f"{prob_lr[1]*100:.1f}%")
                        st.metric("Negative", f"{prob_lr[0]*100:.1f}%")
                
                if model_choice in ["LSTM Neural Network", "Both Models"]:
                    # LSTM prediction
                    sequence = lstm_tokenizer.texts_to_sequences([cleaned_text])
                    padded = pad_sequences(sequence, maxlen=200, padding='post')
                    prob_lstm = lstm_model.predict(padded, verbose=0)[0][0]
                    pred_lstm = 1 if prob_lstm > 0.5 else 0
                    
                    sentiment_lstm = "Positive 😊" if pred_lstm == 1 else "Negative 😞"
                    confidence_lstm = (prob_lstm if pred_lstm == 1 else 1-prob_lstm) * 100
                    
                    if model_choice == "Both Models":
                        st.markdown("---")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("### 🧠 LSTM Neural Network")
                        
                        if pred_lstm == 1:
                            st.markdown(f'<div class="sentiment-positive"><h3>Sentiment: {sentiment_lstm}</h3></div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="sentiment-negative"><h3>Sentiment: {sentiment_lstm}</h3></div>', 
                                      unsafe_allow_html=True)
                        
                        st.progress(confidence_lstm / 100)
                        st.markdown(f"**Confidence:** {confidence_lstm:.2f}%")
                    
                    with col2:
                        st.markdown("### 📈 Probability")
                        st.metric("Positive", f"{prob_lstm*100:.1f}%")
                        st.metric("Negative", f"{(1-prob_lstm)*100:.1f}%")
                
                # Show preprocessed text
                with st.expander("🔍 View Preprocessed Text"):
                    st.text(cleaned_text)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit | Machine Learning & Deep Learning Sentiment Analysis</p>
    <p>Model Performance: LR ~88% | LSTM ~89% Accuracy</p>
</div>
""", unsafe_allow_html=True)
