# 📰 Fake News Detection Web App

## 🚀 Overview

**Fake News Detection** is an interactive web application built using machine learning and NLP techniques to detect whether a news article is **Fake** or **Real**.  
The app offers insightful metrics, sentiment & readability analysis, LIME-based model explanations, and an intuitive UI for both manual and URL-based predictions.

🔗 **Live Demo**: [Fake News Detection App](https://fake-news-detection-prerna.streamlit.app)

---

## ✨ Features

- 🔍 **Single Article Detection**  
  Enter any news content and instantly detect its authenticity.

- 🌐 **URL-Based Prediction**  
  Paste a news article URL — the app scrapes and analyzes the content.

- 📊 **Prediction Output**  
  Displays label (Fake/Real), confidence %, word count, character count, sentiment polarity, and readability score.

- ☁️ **WordCloud Generator**  
  Visualizes frequently occurring terms in the article.

- 🧠 **Explainable AI (LIME)**  
  Highlights important words influencing the model prediction.

- 📉 **Interactive Visualizations**  
  Sentiment graphs, word frequency charts, and readability analysis.

- 🎨 **Modern UI**  
  Built using Streamlit — fast, responsive, and theme-adaptive.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **ML Models**: scikit-learn, XGBoost  
- **NLP Tools**: TextBlob, textstat, WordCloud  
- **Vectorization**: TF-IDF  
- **Model Explainability**: LIME  
- **Data Wrangling**: Pandas, NumPy  
- **Scraping**: BeautifulSoup, requests  
- **Visualizations**: Plotly, Seaborn, Matplotlib  
- **Model Serialization**: Joblib  

---

## 📂 Project Structure
.
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── styles.css # Custom CSS styles
├── fake_news_detection.ipynb # Model training and EDA notebook
├── Datasets/
│ ├── Fake.csv
│ └── True.csv
├── app_pages/ # Modular UI pages
│ ├── How_to_Use.py
│ ├── Tips_for_Spotting_fake_news.py
│ └── About_the_technology.py
├── fake_news_pipeline.joblib # Trained ML pipeline
├── ensemble_model.joblib # Ensemble model
├── lime_config.joblib # LIME configuration
├── tfidf_vectorizer.joblib # TF-IDF vectorizer
└── README.md # Project documentation


---

## 📈 How It Works

1. **Input Text or URL**  
   Users can either type a news article or provide a URL.

2. **Preprocessing & Prediction**  
   Text is cleaned, vectorized, and passed through trained models.

3. **Output Metrics**  
   Model returns label, probability, sentiment score, readability, and more.

4. **Explainability**  
   LIME explains key features (words) influencing the output.

5. **Visuals**  
   Word clouds, sentiment polarity, and other graphs are rendered.

---

## 🏁 Getting Started

### 📦 1. Clone the Repository

```bash
git clone https://github.com/prerna-patil26/fake-news-detection.git
cd fake-news-detection
