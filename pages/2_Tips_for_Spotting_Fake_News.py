import streamlit as st
import os

st.set_page_config(
    page_title="Tips for Spotting Fake News", 
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

st.title("💡 Tips for Spotting Fake News")
st.markdown("""
### 🔍 Common Fake News Indicators:
- **Sensational headlines** with excessive punctuation!!!
- **Unfamiliar websites** or strange URLs
- **Poor grammar** and spelling mistakes
- **Lack of author information** or sources
- **Emotional language** designed to provoke

### 📰 Subject Areas to Watch:
<p><b>Fake News</b> often comes from subjects like:</p>
<ul>
<li>News</li>
<li>Politics</li>
<li>Left-news</li>
<li>Government News</li>
<li>US News</li>
<li>Middle-east</li>
</ul>

<p><b>Reliable News</b> usually comes from:</p>
<ul>
<li>PoliticsNews</li>
<li>WorldNews</li>
<li>Established publications (BBC, Reuters, etc.)</li>
</ul>

### 🛡️ Verification Tools:
- Reverse image search (Google Images)
- Fact-checking sites (Snopes, FactCheck.org)
- Cross-reference with multiple sources
""", unsafe_allow_html=True)

if st.button("⬅️ Back to Main Page"):
    st.switch_page("app.py")