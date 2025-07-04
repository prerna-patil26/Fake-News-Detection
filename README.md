## 📰 Fake News Detection Web App

### 🚀 Overview
# Fake News Detection is a powerful and interactive web application built using machine learning
# and NLP techniques to detect whether a news article is Fake or Real.
# The app includes insightful metrics, sentiment & readability analysis, LIME-based model explanations,
# and an intuitive UI for both manual and URL-based predictions.
# 🔗 Live Demo: https://fake-news-detection-prerna.streamlit.app


## ✨ Features

### 🔍 Single Article Detection
# Input any news text and instantly predict whether it’s fake or real.

### 🌐 URL-Based Prediction
# Paste a news article URL, and the app scrapes and analyzes it automatically.

### 📊 Prediction Output
# Prediction label, confidence %, word count, character count, sentiment polarity, and readability score.

### ☁️ WordCloud Generator
# Generates a word cloud of the most frequent terms in the article.

### 🧠 Explainable AI (LIME)
# Highlights which words most influenced the prediction.

### 📉 Interactive Visualizations
# Charts for text analytics like sentiment, readability, and word frequency.

### 🎨 Modern UI
# Built with Streamlit — fast, responsive, and theme-adaptive design.


## 🛠️ Tech Stack

### Frontend
# Streamlit

### Backend
# Python 3

### ML Models
# scikit-learn, XGBoost

### Data Handling
# Pandas, NumPy

### Visualization
# Plotly, Matplotlib, Seaborn

### Explainability
# LIME

### Text Analytics
# TextBlob, textstat, WordCloud

### Scraping
# BeautifulSoup, requests

### Model Serialization
# Joblib


## 📂 Project Structure

### .
#### ├── app.py
# Main Streamlit app

#### ├── requirements.txt
# Python dependencies

#### ├── styles.css
# Custom CSS styles

#### ├── fake_news_detection.ipynb
# Notebook with experiments & EDA

### Datasets/
#### ├── Fake.csv
# Fake news data

#### ├── True.csv
# Real news data

### app_pages/
#### ├── How_to_Use.py
# Instructions page

#### ├── Tips_for_Spotting_fake_news.py
# Tips for identifying fake news

#### └── About_the_technology.py
# Details about the underlying technology

### pages/ (optional duplicate)
#### ├── How_to_Use.py
# Instructions page

#### ├── Tips_for_Spotting_fake_news.py
# Tips for identifying fake news

#### └── About_the_technology.py
# Details about the underlying technology

#### ├── fake_news_pipeline.joblib
# Trained ML pipeline

#### ├── ensemble_model.joblib
# Ensemble model

#### ├── lime_config.joblib
# LIME configuration

#### ├── tfidf_vectorizer.joblib
# TF-IDF vectorizer

#### └── README.md
# This documentation file


## 📈 How It Works

### Input Text or URL
# Users can type a news article or paste a URL.

### Preprocessing & Prediction
# The app processes the input using a trained ML pipeline and vectorizer.

### Prediction Output
# Displays prediction label, probability, sentiment, readability, etc.

### Explainability
# LIME highlights the top words influencing the result.

### Visuals
# Word cloud, sentiment graphs, readability charts, and more.


## 🏁 Getting Started

### 🔧 Step 1: Clone the Repository
# git clone https://github.com/prerna-patil26/fake-news-detection.git
# cd fake-news-detection

### 📦 Step 2: Install Dependencies
# pip install -r requirements.txt

### ▶️ Step 3: Run the App
# streamlit run app.py

# Then open the app in your browser at http://localhost:8501 or use the Live Demo link.


## 📚 Dataset

### Source
# Fake.csv and True.csv
# Source: Kaggle - Fake and Real News Dataset
# https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset


## 👩‍💻 Author

### About
# Prerna Patil
# 📧 Email: prernapatil2608@gmail.com
# 🎓 MCA Student | 🤖 Data Science & Machine Learning Enthusiast
# 🔗 LinkedIn: https://www.linkedin.com/in/prerna-patil26


## 🙏 Acknowledgements

### Thanks To
# Kaggle – Fake and Real News Dataset
# Streamlit, scikit-learn, LIME, Plotly, XGBoost
# Open-source contributors & the Python community 💙
