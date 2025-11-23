# ML Model Deployment API

## 📌 Overview
Production-ready REST API serving multiple trained ML models using FastAPI.

## 🎯 Features
- Multiple model endpoints (Iris, Wine classification)
- Automatic model loading at startup
- Input validation with Pydantic
- Comprehensive API documentation (Swagger UI)
- Error handling and logging
- Docker support
- Production-ready architecture

## 🛠️ Models Deployed

### 1. Iris Classifier
- **Algorithm**: Random Forest
- **Accuracy**: 97%+
- **Input**: 4 features (sepal/petal measurements)
- **Output**: Iris species (setosa, versicolor, virginica)

### 2. Wine Classifier
- **Algorithm**: Logistic Regression
- **Accuracy**: 98%+
- **Input**: 13 chemical features
- **Output**: Wine type (class_0, class_1, class_2)

## 🚀 Quick Start

### Local Deployment

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run API**
```bash
python api.py
```

3. **Access API**
- Documentation: http://localhost:8000/docs
- Homepage: http://localhost:8000
- Health Check: http://localhost:8000/health

### Docker Deployment

1. **Build Image**
```bash
docker build -t ml-api .
```

2. **Run Container**
```bash
docker run -p 8000:8000 ml-api
```

## 📡 API Endpoints

### GET /
- Homepage with API information

### GET /health
- Health check endpoint
- Returns model loading status

### GET /model-info
- Detailed information about loaded models
- Includes accuracy, features, classes

### POST /predict/iris
- Predict Iris species
- **Input**: JSON with 4 features
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
- **Output**: Prediction with probabilities

### POST /predict/wine
- Predict Wine type
- **Input**: JSON with 13 chemical features
- **Output**: Prediction with probabilities

### GET /stats
- API statistics and uptime

## 🧪 Testing the API

### Using cURL
```bash
# Health check
curl http://localhost:8000/health

# Iris prediction
curl -X POST http://localhost:8000/predict/iris   -H "Content-Type: application/json"   -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Wine prediction
curl -X POST http://localhost:8000/predict/wine   -H "Content-Type: application/json"   -d '{"alcohol": 13.2, "malic_acid": 2.0, "ash": 2.4, "alcalinity_of_ash": 18.0, "magnesium": 100.0, "total_phenols": 2.5, "flavanoids": 2.8, "nonflavanoid_phenols": 0.3, "proanthocyanins": 1.8, "color_intensity": 5.0, "hue": 1.0, "od280_od315": 3.0, "proline": 1000.0}'
```

### Using Python
```python
import requests

# Iris prediction
response = requests.post(
    "http://localhost:8000/predict/iris",
    json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
)
print(response.json())
```
## 🔒 Security Features
- Input validation with Pydantic
- Type checking
- Error handling
- CORS configuration

## 📊 Performance
- Response time: <100ms
- Concurrent requests: Supported
- Model loading: On startup (fast subsequent requests)

## 🚀 Production Deployment

### Cloud Platforms

**AWS (Elastic Beanstalk)**
```bash
eb init -p docker ml-api
eb create ml-api-env
eb deploy
```

**Google Cloud (Cloud Run)**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/ml-api
gcloud run deploy --image gcr.io/PROJECT_ID/ml-api --platform managed
```

**Heroku**
```bash
heroku container:push web --app ml-api
heroku container:release web --app ml-api
```
