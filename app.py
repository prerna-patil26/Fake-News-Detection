from sklearn.base import BaseEstimator
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import textstat
from joblib import load
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import re
from urllib.parse import quote
import os
import shutil
from sklearn.ensemble import VotingClassifier
import datetime

# ----------------------------- SETUP PAGES DIRECTORY ----------------------------- #
os.makedirs("app_pages", exist_ok=True)
os.makedirs("pages", exist_ok=True)

for filename in os.listdir("app_pages"):
    if filename.endswith(".py"):
        src = os.path.join("app_pages", filename)
        dst = os.path.join("pages", filename)
        if not os.path.exists(dst):
            try:
                shutil.copy(src, dst)
            except Exception as e:
                st.error(f"Could not copy {filename}: {str(e)}")

# ----------------------------- UI CONFIG ----------------------------- #
st.set_page_config(
    page_title="Fake News Detection", 
    layout="wide",
    page_icon=":newspaper:"
)

# CSS Styling
st.markdown("""
    <style>
   /* Hide default sidebar navigation and page menu */
    section[data-testid="stSidebar"] > div:nth-child(1) > div > ul,
    div[data-testid="stSidebarNav"] {
        display: none !important;
    }
  /* New style to hide entire sidebar when on subpages */
    .hide-sidebar section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Main content styles */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar button styling */
    .stButton>button {
        width: 100%;
        padding: 10px;
        margin: 5px 0;
        background-color: white !important;
        color: #003366 !important;
        border: none !important;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* Confidence Meter Colors */
    .stPlotlyChart .gauge .bg {
        fill: #f0f2f6;
    }
    .stPlotlyChart .gauge .value {
        fill: #FF0000 !important;
    }
    
    /* Text Analysis Metrics Table */
    .stDataFrame {
        background-color: white !important;
        border: 2px solid #87CEEB !important;
    }
    
    .stDataFrame thead tr th {
        background-color: white !important;
        color: #003366 !important;
        font-weight: bold;
        border-bottom: 2px solid #87CEEB !important;
    }
    
    .stDataFrame tbody tr td {
        color: #003366 !important;
        border-bottom: 1px solid #87CEEB !important;
    }
    
    .stDataFrame tbody tr:nth-child(odd) {
        background-color: #f0f8ff !important;
    }
    
    .stDataFrame tbody tr:nth-child(even) {
        background-color: white !important;
    }
    
    /* News Verification Buttons */
    .source-button {
        display: inline-block;
        padding: 10px 15px;
        margin: 8px;
        background-color: white !important;
        color: #003366 !important;
        border: 2px solid #87CEEB !important;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        text-align: center;
        transition: all 0.3s;
    }
    
    /* Prediction banners */
    .real-news {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin-bottom: 20px;
    }
    
    .fake-news {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin-bottom: 20px;
    }
    
    /* LIME Explanation Box */
    .lime-explanation {
        background-color: white !important;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 2px solid #87CEEB !important;
        color: #003366 !important;
    }
    
    .lime-explanation h4 {
        color: #003366 !important;
    }
    
    .lime-explanation ul {
        color: #003366 !important;
    }
    
    /* Scraped article display box */
    .scraped-article {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        border: 2px solid #87CEEB;
        color: #003366;
        margin-top: 10px;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Hide sidebar when on subpages */
    .sidebar-hidden section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Back button styling */
    .back-button {
        background-color: white !important;
        color: #003366 !important;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
        display: block;
        text-align: center;
        font-weight: bold;
        border: none !important;
    }
    
    /* Feedback section styling */
    .feedback-section {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        border: 2px solid #87CEEB;
        color: #003366;
        margin-top: 20px;
    }
    
    .feedback-title {
        color: #003366;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .feedback-thanks {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------- LOAD MODEL ----------------------------- #
@st.cache_resource
def load_model():
    try:
        pipeline = load('fake_news_pipeline.joblib')
        
        if isinstance(pipeline['model'], VotingClassifier):
            pipeline['model'].voting = 'soft'
        
        lime_config = {
            'explainer_params': {
                'class_names': ['Fake', 'Real'],
                'kernel_width': 25
            },
            'explanation_params': {
                'num_features': 10,
                'num_samples': 5000
            }
        }
        return pipeline['vectorizer'], pipeline['model'], lime_config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        class DummyModel:
            def predict(self, X): return np.random.randint(0, 2, X.shape[0])
            def predict_proba(self, X): return np.array([[0.3, 0.7]] * X.shape[0])
        return Pipeline([('dummy', BaseEstimator())]), DummyModel(), {
            'explainer_params': {
                'class_names': ['Fake', 'Real'],
                'kernel_width': 25
            },
            'explanation_params': {
                'num_features': 10,
                'num_samples': 5000
            }
        }

vectorizer, model, lime_config = load_model()

# ----------------------------- HELPER FUNCTIONS ----------------------------- #
def scrape_article(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        html = requests.get(url, headers=headers).text
        soup = BeautifulSoup(html, 'html.parser')
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return ""

def analyze_text(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
    else:
        proba = [0.3, 0.7]
    
    if prediction == 0:
        confidence = min(proba[0] * 1.2, 0.99)
    else:
        confidence = proba[1]
    
    exp = None
    if hasattr(model, 'predict_proba'):
        try:
            explainer = LimeTextExplainer(**lime_config['explainer_params'])
            exp = explainer.explain_instance(
                text, 
                classifier_fn=lambda x: model.predict_proba(vectorizer.transform(x)),
                **lime_config['explanation_params']
            )
        except Exception as e:
            st.warning(f"Could not generate LIME explanation: {str(e)}")
    
    wc = WordCloud(width=400, height=200, background_color='white').generate(text)
    
    return (
        'Real' if prediction == 1 else 'Fake',
        len(text.split()),
        len(text),
        TextBlob(text).sentiment.polarity,
        textstat.flesch_reading_ease(text),
        wc,
        confidence,
        prediction,
        exp
    )

def handle_chat_query(query):
    # Model-related questions and responses
    model_responses = {
        "how does this model work": [
            "1. Analyzes text patterns using machine learning",
            "2. Trained on verified real and fake news datasets",
            "3. Evaluates multiple linguistic features"
        ],
        "how accurate is this": [
            "1. Accuracy around 85-90% on test data",
            "2. Performance varies by news category",
            "3. Always verify with other sources"
        ],
        "what features does it check": [
            "1. Sentiment and emotional language",
            "2. Readability scores",
            "3. Presence of common fake news markers"
        ],
        "how to spot fake news": [
            "1. Check multiple reputable sources",
            "2. Verify author credentials",
            "3. Look for supporting evidence"
        ],
        "what makes news fake": [
            "1. Misleading headlines",
            "2. Lack of credible sources",
            "3. Emotional manipulation"
        ]
    }
    
    # Suggested model-related questions
    suggested_questions = [
        "How does this model work?",
        "How accurate is the model?",
        "What features does the model check?",
        "How can I spot fake news?",
        "What makes news fake?",
        "Can you explain the confidence score?",
        "What is LIME explanation?",
        "How to verify news authenticity?"
    ]
    
    query_lower = query.lower()
    
    # Check for model-related questions
    for question, response in model_responses.items():
        if question in query_lower:
            return response
    
    # Check for partial matches
    if any(keyword in query_lower for keyword in ["work", "model", "algorithm"]):
        return model_responses["how does this model work"]
    elif any(keyword in query_lower for keyword in ["accurate", "precision", "reliable"]):
        return model_responses["how accurate is this"]
    elif any(keyword in query_lower for keyword in ["feature", "check", "look for"]):
        return model_responses["what features does it check"]
    elif any(keyword in query_lower for keyword in ["spot", "identify", "detect"]):
        return model_responses["how to spot fake news"]
    elif any(keyword in query_lower for keyword in ["fake", "false", "misinformation"]):
        return model_responses["what makes news fake"]
    
    # For unrelated questions
    response = [
        "I specialize in fake news detection. Here are some questions I can help with:",
        "Try asking about:"
    ]
    response.extend([f"- {question}" for question in suggested_questions])
    return response

def save_feedback(text, prediction, confidence, feedback_type, user_comment=""):
    """Save feedback to a CSV file with UTF-8 encoding"""
    feedback_file = "feedback_data.csv"
    
    # Create file if it doesn't exist with UTF-8 encoding
    if not os.path.exists(feedback_file):
        with open(feedback_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,text,prediction,confidence,feedback_type,user_comment\n")
    
    # Escape quotes in text and comments
    text = text.replace('"', "'")
    user_comment = user_comment.replace('"', "'")
    
    # Append feedback with UTF-8 encoding
    with open(feedback_file, 'a', encoding='utf-8') as f:
        f.write(f'"{datetime.datetime.now()}","{text}","{prediction}","{confidence}","{feedback_type}","{user_comment}"\n')
# ----------------------------- SIDEBAR LAYOUT ----------------------------- #
with st.sidebar:
    st.title("🤖 NewsBot Assistant")
    st.write("Hi! I'm NewsBot. I'm here to help you understand predictions and identify fake news.")
    
    st.markdown("---")
    st.markdown("**Quick Actions**")
    
    if st.button("📌 How to Use"):
        st.markdown('<div class="hide-sidebar"></div>', unsafe_allow_html=True)
        st.switch_page("pages/1_How_to_Use.py")
    
    if st.button("💡 Tips for Spotting Fake News"):
        st.markdown('<div class="hide-sidebar"></div>', unsafe_allow_html=True)
        st.switch_page("pages/2_Tips_for_Spotting_Fake_News.py")
    
    if st.button("🔍 About the Technology"):
        st.markdown('<div class="hide-sidebar"></div>', unsafe_allow_html=True)
        st.switch_page("pages/3_About_the_Technology.py")
    
    st.markdown("---")
    
    if st.button("💬 Toggle Chat Assistant"):
        st.session_state.chat_active = not st.session_state.get('chat_active', False)
    
    if st.session_state.get('chat_active', False):
        if user_query := st.text_input("Your question:", key="chat_query"):
            response = handle_chat_query(user_query)
            st.markdown("**Assistant:**")
            for point in response:
                st.markdown(f"- {point}")
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Analysis"):
        st.session_state.clear()
        st.rerun()

# ----------------------------- MAIN APP ----------------------------- #
st.title("Fake News Detection Model")

# Initialize session state variables
if 'input_type_radio' not in st.session_state:
    st.session_state.input_type_radio = "Enter Text"
if 'scraped_text' not in st.session_state:
    st.session_state.scraped_text = ""
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

def clear_inputs():
    if st.session_state.input_type_radio == "Enter Text":
        st.session_state.text_input_area = ""
    else:
        st.session_state.url_input_field = ""
    st.session_state.scraped_text = ""
    st.session_state.feedback_submitted = False

# Input Section
input_col1, input_col2 = st.columns([0.85, 0.15])
with input_col1:
    input_option = st.radio(
        "Input Method:", 
        ["Enter Text", "Enter URL"], 
        horizontal=True, 
        key="input_type_radio"
    )
    
    if input_option == "Enter Text":
        text = st.text_area(
            "Article Text:", 
            height=200, 
            key="text_input_area"
        )
    else:
        url = st.text_input(
            "Article URL:", 
            key="url_input_field"
        )

with input_col2:
    st.write("")  # Spacer
    st.write("")  # Spacer
    st.button("🗑️ Clear", key="clear_btn", help="Clear all inputs", on_click=clear_inputs)

# Always show Analyze button
if st.session_state.input_type_radio == "Enter Text":
    if st.button("🕵️‍♀️ Analyze News", key="analyze_btn", use_container_width=True):
        st.session_state.analyze_clicked = True
        st.session_state.feedback_submitted = False
else:
    if st.button("🕵️‍♀️ Analyze News", key="analyze_btn", use_container_width=True):
        st.session_state.analyze_clicked = True
        st.session_state.feedback_submitted = False

# Get the current input value
text_input = ""
if st.session_state.input_type_radio == "Enter Text":
    text_input = st.session_state.text_input_area if 'text_input_area' in st.session_state else ""
else:
    if st.session_state.scraped_text:
        text_input = st.session_state.scraped_text
    elif 'url_input_field' in st.session_state and st.session_state.url_input_field and st.session_state.analyze_clicked:
        with st.spinner("Scraping article..."):
            scraped_text = scrape_article(st.session_state.url_input_field)
            if scraped_text:
                st.session_state.scraped_text = scraped_text
                text_input = scraped_text
                st.markdown(
                    '<div style="color:#00FF00;font-weight:bold;padding:10px;background-color:#1F2937;border-left:5px solid #00FF00">✅ Article scraped successfully!</div>', 
                    unsafe_allow_html=True
                )
                
                st.markdown("### Scraped Article Content")
                st.markdown(f'<div class="scraped-article">{text_input}</div>', unsafe_allow_html=True)

if st.session_state.analyze_clicked and text_input.strip():
    label, word_count, char_count, sentiment, readability, wc, confidence, prediction, exp = analyze_text(text_input)
    
    if label == "Real":
        st.markdown(f"""
        <div class="real-news">
            <h3>Prediction: 🟢 Real News (Confidence: {confidence*100:.1f}%)</h3>
            <p>This article appears to be credible based on our analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="fake-news">
            <h3>Prediction: 🔴 Fake News (Confidence: {confidence*100:.1f}%)</h3>
            <p>This article shows characteristics of potentially false information.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Results Display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Word Cloud")
        plt.figure(figsize=(6,3))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    with col2:
        st.markdown("#### Confidence Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence*100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': "red" if label == "Fake" else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "#ffcccc"},
                    {'range': [40, 70], 'color': "#fff3cd"},
                    {'range': [70, 100], 'color': "#d4edda"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence*100}
            }))
        st.plotly_chart(fig)
    
    # Metrics Table
    st.markdown("#### Text Analysis Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Word Count", "Character Count", "Sentiment", "Readability"],
        "Value": [word_count, char_count, f"{sentiment:.2f}", f"{readability:.1f}"],
        "Description": [
            "Total words in the article",
            "Total characters in the article",
            "Positive (1) to Negative (-1) sentiment",
            "Higher = easier to read (60-70 is standard)"
        ]
    })
    st.dataframe(metrics_df, use_container_width=True)
    
    # LIME Explanation Section
    if exp is not None:
        st.markdown("---")
        st.markdown("### 🔍 LIME Explanation")
        
        if label == "Real":
            st.markdown("""
            <div class="lime-explanation">
                <h4>How this explanation works:</h4>
                <p>The LIME model highlights words that most influenced the prediction:</p>
                <ul>
                    <li>🟢 <strong style="color:#003366">Green words</strong> support the 'Real News' classification</li>
                    <li>🔴 <strong style="color:#003366">Red words</strong> would support 'Fake News' if present</li>
                </ul>
                <p>Longer bars indicate stronger influence on the prediction.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="lime-explanation">
                <h4>How this explanation works:</h4>
                <p>The LIME model highlights words that most influenced the prediction:</p>
                <ul>
                    <li>🔴 <strong style="color:#003366">Red words</strong> support the 'Fake News' classification</li>
                    <li>🟢 <strong style="color:#003366">Green words</strong> would support 'Real News' if present</li>
                </ul>
                <p>Longer bars indicate stronger influence on the prediction.</p>
            </div>
            """, unsafe_allow_html=True)
        
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(10, 5)
        plt.title("Words Influencing the Prediction", fontsize=14, color='#003366')
        plt.xlabel("Importance", fontsize=12, color='#003366')
        st.pyplot(fig)
        plt.close()
    
    # News Verification
    st.markdown("---")
    st.markdown("### 📰 Verify on News Sites")
    search_query = re.sub(r"\s+", " ", text_input)
    search_query = re.sub(r"[^\w\s]", "", search_query)
    search_query = search_query.strip()  # Keep original text format
    
    news_sites = [
        {"name": "Google News", "url": f"https://news.google.com/search?q={quote(search_query)}", "icon": "🔍"},
        {"name": "ABP News", "url": f"https://news.abplive.com/search/{quote(search_query)}", "icon": "📺"},
        {"name": "BBC News", "url": f"https://www.bbc.co.uk/search?q={quote(search_query)}", "icon": "🇬🇧"},
        {"name": "Reuters", "url": f"https://www.reuters.com/search/news?blob={quote(search_query)}", "icon": "🌐"},
        {"name": "NDTV", "url": f"https://www.ndtv.com/topic/{quote(search_query)}", "icon": "📡"},
        {"name": "Times of India", "url": f"https://timesofindia.indiatimes.com/topic/{quote(search_query)}", "icon": "🗞️"},
        {"name": "The Hindu", "url": f"https://www.thehindu.com/search/?q={quote(search_query)}", "icon": "📰"},
        {"name": "India Today", "url": f"https://www.indiatoday.in/search/{quote(search_query)}", "icon": "🇮🇳"}
    ]
    
    cols = st.columns(3)
    for i, source in enumerate(news_sites):
        with cols[i % 3]:
            st.markdown(f"""
            <a href="{source['url']}" target="_blank" class="source-button">
                <span style="font-size: 1.2em;">{source['icon']}</span> {source['name']}
            </a>
            """, unsafe_allow_html=True)
    
    # ----------------------------- FEEDBACK SECTION ----------------------------- #
    st.markdown("---")
    st.markdown("### 💬 Help Improve Our Model")
    st.markdown("""
    <div class="feedback-section">
        <div class="feedback-title">Was this prediction correct?</div>
        <p>Your feedback helps us improve the model's accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.feedback_submitted:
        with st.form(key='feedback_form'):
            feedback_type = st.radio(
                "Select your feedback:",
                options=[
                    "✅ Prediction was correct",
                    "❌ Prediction was incorrect",
                    "🤔 I'm not sure"
                ],
                key="feedback_radio"
            )
            
            user_comment = st.text_area(
                "Additional comments (optional):",
                height=100,
                key="feedback_comment"
            )
            
            submit_button = st.form_submit_button("Submit Feedback")
            
            if submit_button:
                # Save the feedback
                save_feedback(
                    text=text_input[:1000],  # Store first 1000 chars to avoid huge files
                    prediction=label,
                    confidence=confidence,
                    feedback_type=feedback_type,
                    user_comment=user_comment
                )
                
                st.session_state.feedback_submitted = True
                st.rerun()
    
    if st.session_state.feedback_submitted:
        st.markdown("""
        <div class="feedback-thanks">
            <h4>🙏 Thanks for your feedback!</h4>
            <p>We'll use this to improve our model's accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("↩️ Provide Different Feedback"):
            st.session_state.feedback_submitted = False
            st.rerun()

elif st.session_state.analyze_clicked and not text_input.strip():
    st.warning("Please provide text or URL to analyze")