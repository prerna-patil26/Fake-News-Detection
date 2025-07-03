import streamlit as st
from PIL import Image
import base64

def img_to_bytes(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def show_bio():
    st.set_page_config(page_title="About Me", layout="wide")
    
    # Same title-button layout as technology page
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("About Prerna")
    with col2:
        if st.button("⬅ Back to Technology"):
            st.switch_page("pages/3_About_the_Technology.py")
    
    # Rest of your bio content...
    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            img_html = f"<img src='data:image/png;base64,{img_to_bytes('developer.png')}' width='200' style='border-radius:10px;border:2px solid #87CEEB'>"
            st.markdown(img_html, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Profile image not found: {e}")

    with col2: 
        st.markdown("""
        ### 👋 Hi, I'm Prerna Sharma
        **MCA Student** | AI & ML Enthusiast
        
        📍 **Location:** Your City  
        📧 **Email:** your.email@example.com  
        🔗 **LinkedIn:** [yourprofile](https://linkedin.com/in/yourprofile)  
        💻 **GitHub:** [yourusername](https://github.com/yourusername)
        """)

    # Rest of your bio sections...
    st.markdown("## 🎓 Education")
    st.markdown("""
    - **Master of Computer Applications (MCA)**  
      *Your University* (2022-2024)
    """)

if __name__ == "__main__":
    show_bio()