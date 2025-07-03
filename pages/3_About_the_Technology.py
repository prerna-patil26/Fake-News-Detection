import streamlit as st

def about_technology():
    st.set_page_config(page_title="About the Technology", layout="wide")
    
    # Custom CSS for the button positioning
    st.markdown("""
    <style>
        .title-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .bio-button {
            background-color: #003366 !important;
            color: white !important;
            border: 1px solid white !important;
            border-radius: 5px !important;
            padding: 8px 15px !important;
            font-size: 14px !important;
            cursor: pointer !important;
            text-decoration: none !important;
        }
        .bio-button:hover {
            background-color: #87CEEB !important;
            color: #003366 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title with button in the same line
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("About the Technology")
    with col2:
        if st.button("Know Prerna", key="bio_button"):
            st.switch_page("pages/4_About_Me.py")

st.markdown("""
### 🛠️ NLP Techniques:
We applied a full NLP pipeline to clean and analyze the news articles, which includes:
- **Named Entity Recognition (NER)**: Detects persons, locations, organizations, etc.
- **POS Tagging**: Part-of-Speech tagging helps understand grammatical structure
- **Stopword Removal**
- **Lemmatization**
- **Remove Punctuations**
- **Remove Words with Numbers**
- **Remove Newlines**
- **Lowercasing**
- **Remove Square Brackets**
- **Remove Non-word Characters**
- **Remove URLs**
- **Remove HTML Tags**

---

### 📊 Feature Engineering:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Core vectorization technique
- **Sentiment Score**
- **Readability Metrics**
- **Word Count**
- **Character Count**

---

### 🤖 Machine Learning Models:
We tested multiple models for detecting fake news:
- Logistic Regression
- Decision Tree
- Random Forest
- Passive Aggressive Classifier
- XGBoost
- Naive Bayes

---

### 🧠 Ensemble Model:
- We used a **Hard Voting Classifier** to combine the best models
- Makes final predictions based on majority vote
- Improves reliability and performance

---

### 📈 Performance Metrics:
| Model                    | Accuracy |
|--------------------------|----------|
| Logistic Regression      | 95%      |
| Decision Tree            | 92%      |
| Random Forest            | 96%      |
| Passive Aggressive       | 94%      |
| XGBoost                  | 97%      |
| Naive Bayes              | 91%      |
| **Ensemble (Voting)**    | **98%**  |

- Precision, Recall, and F1-score are computed and visualized
- ROC Curve and Precision-Recall Curve used for evaluation

---

### 🧠 Model Explainability:
- **LIME (Local Interpretable Model-agnostic Explanations)** is used for interpretability
  - 🟢 Green Words: Push prediction towards **Real News**
  - 🔴 Red Words: Push prediction towards **Fake News**
- Users can view **interactive explanations** for every prediction

---

### 📰 Dataset Insights:
- ~50,000 balanced articles from credible and fake sources
- Source examples: **Politifact**, **Reuters**, **BuzzFeed**
- Label-based observation:
  - ❌ Fake News commonly comes from:
    - News, Politics, Left-news, Government News, US News, Middle-east
  - ✅ Real News more likely comes from:
    - PoliticsNews, WorldNews

""", unsafe_allow_html=True)

if st.button("⬅️ Back to Main Page"):
    st.switch_page("app.py")
