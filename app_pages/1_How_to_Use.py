import streamlit as st
import os

st.set_page_config(
    page_title="How to Use",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default navigation
st.markdown("""
<style>
    section[data-testid="stSidebar"] > div:nth-child(1) > div > ul {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("📌 How to Use")
st.markdown("""
### 🚀 Step-by-Step Guide

1. **Enter Content**:
   - Paste full article text **or** input a news URL
   - Use 🗑️ to clear the input

2. **Analyze**:
   - Click **🕵️‍♀️ Analyze** to process the content
   - We'll extract features, run ML models, and generate insights

3. **View Results**:
   - **Prediction**: 🟢 Real or 🔴 Fake with confidence
   - **Metrics**: Word count, sentiment, readability
   - **Visuals**: Word cloud, confidence meter

4. **Verify**:
   - Check source links and highlighted keywords
   - Compare with trusted sources

### 💡 Tips:
- Use full articles (300+ words) for accuracy
- For controversial topics, cross-check with multiple outlets
""", unsafe_allow_html=True)

if st.button("⬅️ Back to Main Page"):
    st.switch_page("app.py")