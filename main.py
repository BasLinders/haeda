import streamlit as st

st.set_page_config(
    page_title="Advanced Exploratory Data Analysis Toolkit",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Main Page UI
logo_url = "https://cdn.homerun.co/49305/hh-woordmerk-rgb-blue-met-discriptor1666785216logo.png"
st.image(logo_url, width=200)

st.title("Advanced Exploratory Data Analysis Toolkit")
st.write("### <span style='color: orange;'>v0.0.1 (beta)</span>", unsafe_allow_html=True)
st.write("""
This is the main page for the Advanced Exploratory Data Analysis Toolkit. You can navigate to individual apps using the sidebar.

### What you're looking at
This toolkit has been created for the purposes of analyzing data from online controlled experiments ('A/B tests') to learn from and better understand user behavior.  

### Features
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Market Basket analysis**: Find frequently bought combinations<br>


### How to Use
- Select a page from the sidebar to view different tools.
- Each page contains a single tool for the purposes described above.

### About
Happy Horizon is a creative digital agency of experts in strategic thinking, analysis, creativity, digital services and technology.
""", unsafe_allow_html=True)

linkedin_url = "https://www.linkedin.com/in/blinders/"
happyhorizon_url = "https://happyhorizon.com/"
footnote_text = f"""Engineered and developed by <a href="{linkedin_url}" target="_blank">Bas Linders</a> @<a href="{happyhorizon_url}" target="_blank">Happy Horizon.</a>"""
st.markdown(footnote_text, unsafe_allow_html=True)