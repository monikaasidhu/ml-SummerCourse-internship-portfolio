
# Sentiment-Based Product Recommendation System (Capstone)

[![Docker Hub](https://img.shields.io/badge/DockerHub-shibani--sentiment--rec-blue)](https://hub.docker.com/r/helloshibani/sentiment-recommendation)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/helloshibani/Sentiment-Based-Product-Recommendation-Analysis)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> 🚀 Capstone project developed by a 4-member team.  
> 🛠 Dockerization and container deployment independently implemented by [Shibani Roychoudhury](https://www.linkedin.com/in/helloshibani/).

---

## 📚 Project Overview

- **Goal**: Build a recommendation system that suggests products to users based on their sentiment expressed in reviews.
- **Approach**: 
  - Built a **Sentiment Analysis Model** using textual features from reviews.
  - Integrated **User-User Collaborative Filtering** for personalized recommendations.
  - Dockerized the full system for consistency across environments (**solo contribution**).

---

## 🛠 Key Highlights

- Used **XGBoost Classifier** with tuned hyperparameters for sentiment classification.
- Focused on **data leakage prevention** and meaningful feature engineering.
- The **Collaborative Filtering** component uses **User-User similarity** to recommend products based on inferred sentiment and user behavior.
- ✅ **Dockerized and deployed** the system using custom scripts.

---

## 🚀 Docker Deployment

- The full system is containerized and can be run with Docker.
- Docker image pushed to **Docker Hub**.
- Deployed briefly on **Railway.app** for public testing.

🔗 **Docker Hub**: [helloshibani/sentiment-recommendation](https://hub.docker.com/r/helloshibani/sentiment-recommendation)

> 🛠 **Dockerization independently contributed by Shibani Roychoudhury**.  
> 📄 See [docker_notes.md](docker_notes.md) for build/run instructions.

---

## 🛠 Tech Stack

- **Python**  
- **Pandas**, **NumPy**, **XGBoost**
- **FastAPI + Jinja2**
- **Docker**, **Uvicorn**
- **Railway (optional deployment)**

---

## 📈 How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/helloshibani/Sentiment-Based-Product-Recommendation-Analysis.git
    cd Sentiment-Based-Product-Recommendation-Analysis
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Build and run the Docker container:
    ```bash
    docker build -t sentiment-recommendation-system .
    docker run -p 8000:8000 sentiment-recommendation-system
    ```

Then open `http://localhost:8000` in your browser.

---

## 📁 Folder Structure

- `Sentiment_Recommendation_Capstone.ipynb`: Main notebook
- `app.py`: FastAPI app logic
- `model.py`: Sentiment + recommendation engine
- `templates/`: Contains Jinja2 HTML files
- `pickle/`: Trained models (excluded in repo)
- `Dockerfile`, `Procfile`, `requirements.txt`, `runtime.txt`: For Docker & Railway deployment

---

## 📊 Results and Insights

- Predicts sentiment from user reviews and recommends relevant products.
- Containerized setup allows reproducible testing across platforms.

---

## 🚧 Future Work

- Improve model robustness and runtime performance.
- Streamline model loading in Docker containers.
- Link with [Revised Solo Repo](https://github.com/helloshibani/Sentiment-Based-Product-Recommendation-Analysis-Revision) for enhanced model and improved explainability.

---

## 🙌 Acknowledgements

- Thanks to the Capstone team for collaborative effort and Railway deployment.
- **Docker setup and deployment independently implemented by [Shibani Roychoudhury](https://www.linkedin.com/in/helloshibani/)**

---

## 🔗 Connect

📬 [LinkedIn – Shibani Roychoudhury](https://www.linkedin.com/in/helloshibani/)  
Let’s build, break, and learn together 🚀
