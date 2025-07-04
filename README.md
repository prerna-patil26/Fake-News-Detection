# 📰 Fake News Detection Web App

## 🚀 Project Overview

**Fake News Detection** is a powerful and interactive web application built using **Machine Learning** and **NLP techniques** to classify whether a news article is *Fake* or *Real*.  

The app offers real-time predictions, URL-based article analysis, LIME-based interpretability, sentiment analysis, and an engaging user experience using **Streamlit**.

🔗 **Live Demo**: [Fake News Detection App](https://fake-news-detection-prerna.streamlit.app)

---

## ✨ Key Features

- 🔍 **Single Article Detection**:  
  Enter any news content and get instant classification as fake or real.

- 🌐 **URL-Based Prediction**:  
  Paste a news article URL – the app scrapes and analyzes the content automatically.

- 📊 **Prediction Output**:  
  Includes label, confidence percentage, word & character count, sentiment polarity, and readability score.

- ☁️ **WordCloud Generator**:  
  Visualizes the most frequent terms in the article for better context.

- 🧠 **Explainable AI (LIME)**:  
  Highlights influential words that guided the model’s prediction.

- 📉 **Interactive Visualizations**:  
  Explore sentiment, word frequency, and readability via graphs and charts.

- 🎨 **Modern UI**:  
  Built using Streamlit with a responsive, theme-adaptive interface.

---

## 🛠 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **ML Models**: scikit-learn, XGBoost  
- **Text Analytics**: TextBlob, textstat, WordCloud  
- **Vectorization**: TF-IDF  
- **Explainability**: LIME  
- **Visualization**: Plotly, Seaborn, Matplotlib  
- **Data Wrangling**: Pandas, NumPy  
- **Web Scraping**: BeautifulSoup, requests  
- **Model Serialization**: Joblib

---

## 📂 Project Structure

.
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── styles.css # Custom CSS styles
├── fake_news_detection.ipynb # Jupyter notebook with EDA and experiments
│
├── Datasets/
│ ├── Fake.csv # Fake news dataset
│ └── True.csv # Real news dataset
│
├── app_pages/ # Streamlit pages
│ ├── How_to_Use.py # App instructions
│ ├── Tips_for_Spotting_fake_news.py # Fake news identification tips
│ └── About_the_technology.py # Technology explanations
│
├── models/ # Serialized models
│ ├── fake_news_pipeline.joblib # Main ML pipeline
│ ├── ensemble_model.joblib # Ensemble model
│ ├── lime_config.joblib # LIME explainer config
│ └── tfidf_vectorizer.joblib # TF-IDF vectorizer
│
└── README.md # Project documentation




---

## 📈 How It Works

1. **Input Text or URL**  
   Users can manually input news content or paste a news article link.

2. **Preprocessing & Prediction**  
   The input is preprocessed, vectorized, and passed into a trained ML model.

3. **Output Results**  
   The app returns the classification label (Fake/Real), confidence %, sentiment score, and more.

4. **Explanation with LIME**  
   LIME visually highlights the most influential words in the article that affected the prediction.

5. **Visualizations**  
   Word clouds, sentiment graphs, readability scores, and text analytics are displayed interactively.

---

## 🏁 Getting Started

Follow the steps below to clone and run the project locally:

### 📦 Step 1: Clone the Repository

```bash
git clone https://github.com/prerna-patil26/fake-news-detection.git
cd fake-news-detection



