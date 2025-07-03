import streamlit as st

st.set_page_config(
    page_title="About the Technology", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default navigation from sidebar
st.markdown("""
<style>
    section[data-testid="stSidebar"] > div:nth-child(1) > div > ul {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 About the Technology")

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
