## 📰 Fake News Detection Web App

## 🚀 Overview
# Fake News Detection is a powerful and interactive web application built using machine learning and NLP techniques
# to detect whether a news article is Fake or Real.
# The app includes insightful metrics, sentiment and readability analysis, LIME-based model explanations,
# and an intuitive UI for both manual and URL-based predictions.

# 🔗 Live Demo: https://fake-news-detection-prerna.streamlit.app


## ✨ Features
# 🔍 Single Article Detection: Input any news text and instantly predict whether it’s fake or real.
# 🌐 URL-Based Prediction: Paste a news article URL, and the app scrapes and analyzes it automatically.
# 📊 Prediction Output: Prediction label, confidence %, word count, character count, sentiment, readability.
# ☁️ WordCloud Generator: Generates a word cloud of the most frequent terms in the article.
# 🧠 Explainable AI (LIME): Highlights which words most influenced the prediction.
# 📉 Interactive Visualizations: Sentiment, readability, and word frequency charts.
# 🎨 Modern UI: Built with Streamlit — fast, responsive, theme-adaptive.


## 🛠️ Tech Stack
# Frontend: Streamlit
# Backend: Python 3
# ML Models: scikit-learn, XGBoost
# Data Handling: Pandas, NumPy
# Visualization: Plotly, Matplotlib, Seaborn
# Explainability: LIME
# Text Analytics: TextBlob, textstat, WordCloud
# Scraping: BeautifulSoup, requests
# Model Serialization: Joblib


## 📂 Project Structure
# .
# ├── app.py                         # Main Streamlit app
# ├── requirements.txt              # Python dependencies
# ├── styles.css                    # Custom CSS styles
# ├── fake_news_detection.ipynb    # Notebook with experiments & EDA
# ├── Datasets/
# │   ├── Fake.csv
# │   └── True.csv
# ├── app_pages/
# │   ├── How_to_Use.py
# │   ├── Tips_for_Spotting_fake_news.py
# │   └── About_the_technology.py
# ├── pages/                        # (Duplicate, optional — remove if unused)
# │   ├── How_to_Use.py
# │   ├── Tips_for_Spotting_fake_news.py
# │   └── About_the_technology.py
# ├── fake_news_pipeline.joblib     # Trained ML pipeline
# ├── ensemble_model.joblib         # Ensemble model
# ├── lime_config.joblib            # LIME configuration
# ├── tfidf_vectorizer.joblib       # TF-IDF vectorizer
# └── README.md                     # This documentation


## 📈 How It Works
# 1️⃣ Input Text or URL: Type a news article or paste a URL.
# 2️⃣ Preprocessing & Prediction: Processes input using trained ML pipeline & vectorizer.
# 3️⃣ Prediction Output: Displays label, probability, sentiment, readability etc.
# 4️⃣ Explainability: LIME highlights top words influencing the result.
# 5️⃣ Visuals: Word cloud, sentiment graphs, readability charts, etc.


## 🏁 Getting Started

## 🔧 1. Clone the Repository
# git clone https://github.com/prerna-patil26/fake-news-detection.git
# cd fake-news-detection

## 📦 2. Install Dependencies
# pip install -r requirements.txt

## ▶️ 3. Run the App
# streamlit run app.py

# Open the app in your browser at http://localhost:8501 or use the Live Demo link above.


## 📚 Dataset
# - Fake.csv and True.csv
# - Source: Kaggle - Fake and Real News Dataset
#   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset


## 👩‍💻 Author
# Prerna Patil
# 📧 Email: prernapatil2608@gmail.com
# 🎓 MCA Student | 🤖 Data Science & Machine Learning Enthusiast
# 🔗 LinkedIn: https://www.linkedin.com/in/prerna-patil26


## 🙏 Acknowledgements
# - Kaggle – Fake and Real News Dataset
# - Streamlit, scikit-learn, LIME, Plotly, XGBoost
# - Open-source contributors & the Python community 💙
