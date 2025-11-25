
# Sentiment-Based Product Recommendation System (Capstone)

[![Docker Hub](https://img.shields.io/badge/DockerHub-shibani--sentiment--rec-blue)](https://hub.docker.com/r/helloshibani/sentiment-recommendation)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/helloshibani/Sentiment-Based-Product-Recommendation-Analysis)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---


##  Project Overview
- **Goal**: Build a recommendation system that suggests products to users based on their sentiment expressed in reviews.
- **Approach**: 
  - Built a **Sentiment Analysis Model** using textual features from reviews.
  - Integrated **User-User Collaborative Filtering** for personalized recommendations.
  - Dockerized the full system for consistency across environments (**solo contribution**).
---

##  Key Highlights

- Used **XGBoost Classifier** with tuned hyperparameters for sentiment classification.
- Focused on **data leakage prevention** and meaningful feature engineering.
- The **Collaborative Filtering** component uses **User-User similarity** to recommend products based on inferred sentiment and user behavior.
-  **Dockerized and deployed** the system using custom scripts.

---

##  Docker Deployment

- The full system is containerized and can be run with Docker.
- Docker image pushed to **Docker Hub**.
- Deployed briefly on **Railway.app** for public testing.

---

##  Tech Stack

- **Python**  
- **Pandas**, **NumPy**, **XGBoost**
- **FastAPI + Jinja2**
- **Docker**, **Uvicorn**
- **Railway (optional deployment)**

---


##  Folder Structure

- `Sentiment_Recommendation_Capstone.ipynb`: Main notebook
- `app.py`: FastAPI app logic
- `model.py`: Sentiment + recommendation engine
- `templates/`: Contains Jinja2 HTML files
- `pickle/`: Trained models (excluded in repo)
- `Dockerfile`, `Procfile`, `requirements.txt`, `runtime.txt`: For Docker & Railway deployment

---

##  Results and Insights

- Predicts sentiment from user reviews and recommends relevant products.
- Containerized setup allows reproducible testing across platforms.

---

##  Future Work

- Improve model robustness and runtime performance.
- Streamline model loading in Docker containers.
---



