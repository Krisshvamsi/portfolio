import streamlit as st
import streamlit.components.v1 as components
from forms.contact import contact_form
from pathlib import Path


@st.experimental_dialog("Contact Me")
def show_contact_form():
    contact_form()


# --- HERO SECTION ---
col1, col2 = st.columns([0.3,0.7], gap="small", vertical_alignment="center")
with col1:
    st.image("./assets/6.png", width=250)

with col2:

    st.title("Hello👋...I'm Krishna Vamsi Rokkam", anchor=False)
    st.write("**🧠 AI Innovator | 👨🏻‍💻 Data Scientist  | 🤖 ML Engineer**")
    
    st.html("""<a href="https://www.linkedin.com/in/krishna-vamsi-rokkam/">
<img src="https://img.icons8.com/?size=100&id=13930&format=png&color=000000" alt="LinkedIn Profile" style="width:42px;height:42px;">
</a>  |  
<a href="https://github.com/Krisshvamsi?tab=repositories/">
<img src="https://icon-library.com/images/github-icon-white/github-icon-white-6.jpg" alt="GitHub Profile" style="width:42px;height:42px;">
</a>""")
        
    if st.button("✉️ Contact Me"):
            show_contact_form()

st.write("\n")
st.write("\n")

# components.html("""<hr style="height:5px;border:none;color:#d33682;background-color:#333;" /> """)
# Education

st.subheader("Education", anchor=False)
st.write("""
         - **Master of Applied Computer Science - Concordia University, Canada** (September 2022 - April 2024)
         - **B. Tech in Information Technology - Sreenidhi Institute of Science and Technology, India** (August 2018 - July 2022)
         """)

# Skills
st.subheader("Skills", anchor=False)
st.write("""
- ***Programming languages***: C, C#, Prolog, Python, R, Java, SQL, NoSQL, HTML, CSS, JavaScript
- ***Frameworks & Libraries***: Git, Linux, PyTorch, Hugging Face, Transformers, LLM Fine Tuning, Keras, TensorFlow, NumPy, Pandas, Matplotlib, Tableau, Power BI, R-Studio, plotly, spaCy, Seaborn, nltk, Scikit-learn, Docker, Couchbase, Kubernetes, ElasticSearch, Flask, gRPC, Azure Open AI, AWS Sagemaker, SpeechBrain, mlflow
""")



# Work Experience
st.subheader("Work Experience", anchor=False)

# Microsoft + Nuance
with st.expander("**Microsoft + Nuance, Speech and Data Science Intern** (May - July, 2023)"):
    st.write("""
- Engineered a Python-based Regex Data Redaction Tool to safeguard sensitive PII and PCI data. Integrated gRPC Calls for seamless functionality and rigorously tested each feature through comprehensive test cases in Python.
- Actively contributed to the development of an NLU Data Augmentation Tool utilizing GPT-3/4 to generate training data. Conducted benchmarking, optimized performance through hyperparameter tuning, and analyzed reports generated using Nuance Mix tool.
""")


# Deloitte
with st.expander("**Deloitte, Data Science Intern** (January - August, 2022)"):
    st.write("""
- Acquired proficiency in AI development within the Deloitte Application Studio, specializing in Microsoft Azure services, including LUIS, QnA Maker, TTS, and Bot services.
- Developed a Conversational AI bot using LUIS and QnA Maker services, enhancing user interactions for Audit-related tasks and user stories. Additionally, implemented TTS audio capabilities for the bot using Azure AI speech services.
""")


# Technocolabs Softwares Inc
with st.expander("**Technocolabs Softwares Inc, Data Science Intern** (May - August, 2021)"):
    st.write("""
- Collaborated on a Time Series Forecasting project, specializing in data augmentation techniques to enhance the dataset for improved model performance.
- Developed predictive models to forecast next-second price movements in the stock market, including Logistic Regression, SVM variants, LSTM, and led data visualization and storytelling efforts to effectively communicate project progress and insights to stakeholders.
""")

# SmartBridge Pvt Ltd
with st.expander("**SmartBridge Pvt Ltd, ML Engineer Intern** (May - August, 2020)"):
    st.write("""
- Led the development of a CNN-based deep learning model, handling data preprocessing and successful deployment on the IBM cloud platform.
- Utilized advanced deep learning and optimization techniques, consistently surpassing project goals and earning recognition as the top intern team in RSIP-2020.
""")