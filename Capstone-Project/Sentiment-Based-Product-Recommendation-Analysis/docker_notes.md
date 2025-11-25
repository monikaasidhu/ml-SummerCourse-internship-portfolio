
# 🐳 Docker Notes – Sentiment-Based Product Recommendation System

This document details the Docker-based deployment pipeline built for this project.  
> 🚀 **Dockerization independently contributed by [Shibani Roychoudhury](https://www.linkedin.com/in/helloshibani/)** as part of the Capstone team project.

---

## 📁 Folder Structure

Here’s how the project is structured to support containerized deployment:

```
├── app.py                  # FastAPI app with web + API routes
├── model.py                # Recommendation + sentiment logic
├── Dockerfile              # Docker image definition
├── requirements.txt        # All dependencies
├── Procfile / runtime.txt  # For Railway/Heroku deployment
├── templates/              # HTML UI (Jinja2)
│   └── index.html
├── pickle/                 # Pre-trained models + vectorizers
```

---

## 🔧 What the Docker Image Contains

The image created from this project includes:

- **Python 3.11 base**
- **FastAPI + Uvicorn** server
- Your full application code
- Mounted `templates/` and `pickle/` folders

---

## ⚙️ Build & Run Commands

To run the app locally using Docker:

```bash
# Step 1: Clone the repository
git clone https://github.com/helloshibani/Sentiment-Based-Product-Recommendation-Analysis.git
cd Sentiment-Based-Product-Recommendation-Analysis

# Step 2: Build the image
docker build -t sentiment-recommendation-system .

# Step 3: Run the container
docker run -p 8000:8000 sentiment-recommendation-system
```

> The app will be accessible at: `http://localhost:8000`

---

## 📦 Notes on Model Files

The application depends on multiple `.pkl` files stored inside the `pickle/` directory:

| Filename | Description |
|----------|-------------|
| `user_final_rating.pkl` | Matrix of user-product ratings |
| `cleaned-data.pkl`      | Final cleaned product reviews |
| `tfidf-vectorizer.pkl`  | TF-IDF model for review text |
| `sentiment-classification-xg-boost-best-tuned.pkl` | Trained XGBoost classifier |

⚠️ These files are **not included in this public repo** to avoid large file bloat.  
To use them:
- You may request access from the author (via [LinkedIn](https://www.linkedin.com/in/helloshibani/))
- Or clone the [Revised Solo Version](https://github.com/helloshibani/Sentiment-Based-Product-Recommendation-Analysis-Revision) which includes a reworked model and Docker setup.

---

## 🛠 Deployment Support

- Designed for both **local use** and **Railway/Heroku** cloud deployment.
- `Procfile` and `runtime.txt` included to enable smooth deployment.

---

## 🙌 Credits

This Docker setup was built and tested independently by **Shibani Roychoudhury** as part of her hands-on DevOps learning.

For feedback, improvements, or collaboration:  
📬 [Connect with me](https://www.linkedin.com/in/helloshibani/) on LinkedIn

---
