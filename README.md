# 📰 Fake News Detection Web App

## 🚀 Overview

**Fake News Detection** is a powerful and interactive web application built using machine learning and NLP techniques to detect whether a news article is **Fake** or **Real**.  
The app includes insightful metrics, sentiment and readability analysis, LIME-based model explanations, and an intuitive UI for both manual and URL-based predictions.

🔗 **Live Demo**: [Fake News Detection App](https://fake-news-detection-prerna.streamlit.app)

---

## ✨ Features

- 🔍 **Single Article Detection**  
  Input any news text and instantly predict whether it’s fake or real.

- 🌐 **URL-Based Prediction**  
  Paste a news article URL, and the app scrapes and analyzes it automatically.

- 📊 **Prediction Output**  
  Includes prediction label, confidence %, word count, character count, sentiment polarity, and readability score.

- ☁️ **WordCloud Generator**  
  Generates a word cloud of the most frequent terms in the article.

- 🧠 **Explainable AI (LIME)**  
  Highlights which words most influenced the prediction.

- 📉 **Interactive Visualizations**  
  Shows charts for text analytics like sentiment, readability, and word frequency.

- 🎨 **Modern UI**  
  Built with Streamlit — fast, responsive, and theme-adaptive design.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3
- **ML Models**: scikit-learn, XGBoost
- **Data Handling**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Explainability**: LIME
- **Text Analytics**: TextBlob, textstat, WordCloud
- **Scraping**: BeautifulSoup, requests
- **Model Serialization**: Joblib

---

# 📂 Project Structure
 .
 ├── app.py                         # Main Streamlit app
 ├── requirements.txt              # Python dependencies
 ├── styles.css                    # Custom CSS styles
 ├── fake_news_detection.ipynb    # Notebook with experiments & EDA
 ├── Datasets/
 │   ├── Fake.csv
 │   └── True.csv
 ├── app_pages/                    # Modular Streamlit pages
 │   ├── How_to_Use.py
 │   ├── Tips_for_Spotting_fake_news.py
 │   └── About_the_technology.py
 ├── pages/                        # (Duplicate, optional — remove if unused)
 │   ├── How_to_Use.py
 │   ├── Tips_for_Spotting_fake_news.py
 │   └── About_the_technology.py
 ├── fake_news_pipeline.joblib     # Trained ML pipeline
 ├── ensemble_model.joblib         # Ensemble model
 ├── lime_config.joblib            # LIME configuration
 ├── tfidf_vectorizer.joblib       # TF-IDF vectorizer
 └── README.md                     # This documentation



---

## 📈 How It Works

1. **Input Text or URL**  
   Users can type a news article or paste a URL.

2. **Preprocessing & Prediction**  
   The app processes the input using a pre-trained ML pipeline and vectorizer.

3. **Prediction Output**  
   Displays prediction label, probability, sentiment, and other metrics.

4. **Explainability**  
   LIME highlights the top words influencing the result.

5. **Visuals**  
   Word cloud, sentiment graphs, readability charts, and more.

---

## 🏁 Getting Started

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/prerna-patil26/fake-news-detection.git
cd fake-news-detection

# 📦 2. Install Dependencies
```bash
# pip install -r requirements.txt
# ```

# ▶️ 3. Run the App
# ```bash
# streamlit run app.py
# ```

# 📚 Dataset
# - Fake.csv and True.csv  
#   Source: Kaggle - Fake and Real News Dataset

# 👩‍💻 Author
# **Prerna Patil**  
# 📧 Email: prernapatil2608@gmail.com  
# 🎓 MCA Student | 🤖 Data Science & Machine Learning Enthusiast  
# 🔗 [LinkedIn](https://www.linkedin.com/in/prerna-patil26)

# 🙏 Acknowledgements
# - Kaggle – Fake and Real News Dataset
# - Streamlit, scikit-learn, LIME, Plotly, XGBoost
# - Open-source contributors and the Python community 💙
