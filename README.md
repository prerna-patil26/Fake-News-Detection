# 📰 Fake News Detection Web App

## 🚀 Overview

**Fake News Detection** is a powerful and interactive web application built using **Machine Learning** and **Natural Language Processing (NLP)** techniques to detect whether a news article is *Fake* or *Real*.

The app includes insightful metrics, sentiment & readability analysis, **LIME-based model explanations**, and an intuitive UI for both **manual text input** and **URL-based predictions**.

🔗 **Live Demo**: [Fake News Detection App](https://fake-news-detection-prerna.streamlit.app)


---

## ✨ Features

- 🔍 **Single Article Detection** – Input any news content and instantly detect its authenticity.
- 🌐 **URL-Based Prediction** – Paste a news article URL — the app scrapes and analyzes the content.
- 📊 **Prediction Output** – Displays label (Fake/Real), confidence %, word count, character count, sentiment polarity, and readability score.
- ☁️ **WordCloud Generator** – Visualizes frequently occurring terms in the article.
- 🧠 **Explainable AI (LIME)** – Highlights important words influencing the model prediction.
- 📉 **Interactive Visualizations** – Sentiment graphs, word frequency charts, and readability analysis.
- 🎨 **Modern UI** – Built using **Streamlit** — fast, responsive, and theme-adaptive.

---

## 🛠 Tech Stack

| Layer      | Technologies Used |
|------------|-------------------|
| Frontend   | Streamlit         |
| Backend    | Python            |
| ML Models  | Scikit-learn, XGBoost |
| Text Analytics | TextBlob, textstat, WordCloud |
| Vectorization | TF-IDF         |
| Explainability | LIME          |
| Data Handling | Pandas, NumPy  |
| Scraping   | BeautifulSoup, requests |
| Visualizations | Plotly, Seaborn, Matplotlib |
| Serialization | Joblib         |

---

## 📂 Project Structure

```bash
Fake-News-Detection/
│
├── app.py                        # Main Streamlit app
├── requirements.txt             # Python dependencies
├── styles.css                   # Custom CSS styles
├── fake_news_detection.ipynb    # Model training and EDA notebook
│
├── Datasets/
│   ├── Fake.csv                 # Fake news data
│   └── True.csv                 # Real news data
│
├── app_pages/                   # Modular UI pages
│   ├── How_to_Use.py
│   ├── Tips_for_Spotting_fake_news.py
│   └── About_the_technology.py
│
├── pages/                       # (Optional duplicate UI pages)
│   ├── How_to_Use.py
│   ├── Tips_for_Spotting_fake_news.py
│   └── About_the_technology.py
│
├── fake_news_pipeline.joblib    # Trained ML pipeline
├── ensemble_model.joblib        # Ensemble model
├── lime_config.joblib           # LIME configuration
├── tfidf_vectorizer.joblib      # TF-IDF vectorizer
│
└── README.md                    # Project documentation



```


## 📈 How It Works

1. **Input Text or URL**  
   Users can type a news article or paste a URL of an online article.

2. **Preprocessing & Prediction**  
   The app cleans and transforms the input text, then predicts using trained models.

3. **Output Metrics**  
   Displays prediction label, confidence %, sentiment polarity, readability score, word and character counts.

4. **Explainability**  
   Uses LIME to visualize important words influencing the prediction.

5. **Visuals**  
   Includes WordClouds, sentiment charts, and text statistics for better interpretation.



## 🏁 Getting Started

### 🔧 Step 1: Clone the Repository

```bash
git clone https://github.com/prerna-patil26/fake-news-detection.git
cd fake-news-detection


```

### 📦 Step 2: Install Dependencies

Make sure you have Python installed (preferably Python 3.7+), then install all required libraries using:

```bash
pip install -r requirements.txt

```

### ▶️ Step 3: Launch the App

Start the Streamlit app using the following command:

```bash
streamlit run app.py

```

## 📚 Dataset

This project uses a labeled dataset of real and fake news articles for training and evaluation.

### 🗂 Files

- `Fake.csv` – Contains thousands of fake news articles.
- `True.csv` – Contains thousands of real news articles.




## 👩‍💻 Author

**Prerna Patil**  
🎓 MCA Student | 🤖 Data Science & Machine Learning Enthusiast  
📧 Email: [prernapatil2608@gmail.com](mailto:prernapatil2608@gmail.com)  
🔗 [LinkedIn Profile](#) <!-- Replace with your actual LinkedIn profile link -->




## ⭐ Show Your Support

If you found this project helpful or interesting:

- Give it a ⭐ on [GitHub](https://github.com/prerna-patil26/fake-news-detection)  
- Share it with others  
- Use it in your own projects (with credit)  
- Connect on LinkedIn!

It really helps and means a lot 🙌




