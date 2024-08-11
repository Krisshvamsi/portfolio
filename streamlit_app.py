import streamlit as st

# --- PAGE SETUP ---

st.set_page_config(
    page_title="Krishna's Portfolio",
    layout="wide")


about_page = st.Page(
    "views/about_me.py",
    title="About Me",
    icon="🤖",
    default=True,
)
project_1_page = st.Page(
    "views/projects.py",
    title="My projects",
    icon=":material/bar_chart:",
)
project_2_page = st.Page(
    "views/chatbot.py",
    title="Ask me Anything about Krishna",
    icon=":material/smart_toy:",
)

project_3_page = st.Page(
    "views/rag.py",
    title="Retrieval Augmented Generation Engine",
    icon=":material/smart_toy:",
)



# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [about_page],
        "Projects": [project_1_page, project_2_page, project_3_page],
    }
)


# --- SHARED ON ALL PAGES ---
st.logo("assets/menu.gif")
st.sidebar.text("Made with ❤️ by Krishna ")


# --- RUN NAVIGATION ---
pg.run()