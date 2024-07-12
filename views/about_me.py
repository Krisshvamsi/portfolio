import streamlit as st

from forms.contact import contact_form

@st.experimental_dialog("Contact Me")
def show_contact_form():
    contact_form()
    


# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
with col1:
    st.image("./assets/pic.png", width=230)

with col2:
    st.title("Krishna Vamsi Rokkam", anchor=False)
    st.write("[krishnavamsirokkam2001@gmail.com](mailto:krishnavamsirokkam2001@gmail.com) ")
    st.write("[LinkedIn](https://www.linkedin.com/in/krishna-vamsi-rokkam/) | [GitHub](https://github.com/Krisshvamsi?tab=repositories)")
    st.write(
        "Senior Data Analyst, assisting enterprises by supporting data-driven decision-making."
    )
    if st.button("✉️ Contact Me"):
        show_contact_form()



# Education
st.header("Education")
st.write("**Master of Applied Computer Science - Concordia University, Canada**")
st.write("Sep 2022 - April 2024")
st.write("**B. Tech in Information Technology - Sreenidhi Institute of Science and Technology, India**")
st.write("Aug 2018 - Jul 2022")

# Skills
st.header("Skills")
st.write("""
- **Programming languages**: C, C#, Prolog, Python, R, Java, SQL, NoSQL, HTML, CSS, JavaScript
- **Frameworks & Libraries**: Git, Linux, PyTorch, Hugging Face, Transformers, LLM Fine Tuning, Keras, TensorFlow, NumPy, Pandas, Matplotlib, Tableau, Power BI, R-Studio, plotly, spaCy, Seaborn, nltk, Scikit-learn, Docker, Couchbase, Kubernetes, ElasticSearch, Flask, gRPC, Azure Open AI, AWS Sagemaker, SpeechBrain, mlflow
""")

# Work Experience
st.header("Work Experience")

# Microsoft + Nuance
st.subheader("Microsoft + Nuance, Speech and Data Science Intern")
st.write("**May - Jul, 2023**")
st.write("""
- Engineered a Python-based Regex Data Redaction Tool to safeguard sensitive PII and PCI data. Integrated gRPC Calls for seamless functionality and rigorously tested each feature through comprehensive test cases in Python.
- Actively contributed to the development of an NLU Data Augmentation Tool utilizing GPT-3/4 to generate training data. Conducted benchmarking, optimized performance through hyperparameter tuning, and analyzed reports generated using Nuance Mix tool.
""")

# Deloitte
st.subheader("Deloitte, Data Science Intern")
st.write("**Jan - Aug, 2022**")
st.write("""
- Acquired proficiency in AI development within the Deloitte Application Studio, specializing in Microsoft Azure services, including LUIS, QnA Maker, TTS, and Bot services.
- Developed a Conversational AI bot using LUIS and QnA Maker services, enhancing user interactions for Audit-related tasks and user stories. Additionally, implemented TTS audio capabilities for the bot using Azure AI speech services.
""")

# Technocolabs Softwares Inc
st.subheader("Technocolabs Softwares Inc, Data Science Intern")
st.write("**May - Aug, 2021**")
st.write("""
- Collaborated on a Time Series Forecasting project, specializing in data augmentation techniques to enhance the dataset for improved model performance.
- Developed predictive models to forecast next-second price movements in the stock market, including Logistic Regression, SVM variants, LSTM, and led data visualization and storytelling efforts to effectively communicate project progress and insights to stakeholders.
""")

# SmartBridge Pvt Ltd
st.subheader("SmartBridge Pvt Ltd, ML Engineer Intern")
st.write("**May - Aug, 2020**")
st.write("""
- Led the development of a CNN-based deep learning model, handling data preprocessing and successful deployment on the IBM cloud platform.
- Utilized advanced deep learning and optimization techniques, consistently surpassing project goals and earning recognition as the top intern team in RSIP-2020.
""")