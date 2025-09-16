import streamlit as st

st.set_page_config(
    page_title="Advanced Exploratory Data Analysis",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Main Page UI
logo_url = "https://cdn.homerun.co/49305/hh-woordmerk-rgb-blue-met-discriptor1666785216logo.png"
st.image(logo_url, width=200)

st.title("Advanced Exploratory Data Analysis")
st.write("### <span style='color: orange;'>v0.0.1 (beta)</span>", unsafe_allow_html=True)
st.write("""
This is the main page for the Advanced Exploratory Data Analysis Toolkit. You can navigate to individual apps using the sidebar.

### What you're looking at
This collection of tools makes it easy to analyze and visualize data from various sources, without the need for complex coding or data manipulation. Each tool is designed to help you quickly gain insights from your data.

### Features
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Market Basket analysis**: Find frequently bought combinations on product / category level<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Process mining**: Reconstruct the customer journey (in development)<br>

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
