import streamlit as st
import streamlit.components.v1 as components
from forms.contact import contact_form
from pathlib import Path
import time


@st.experimental_dialog("Contact Me")
def show_contact_form():
    contact_form()


# --- HERO SECTION ---
col1, col2 = st.columns([0.3, 0.7], gap="small", vertical_alignment="center")
with col1:
    st.image("./assets/6.png", width=250)

with col2:
    st.title("Hello👋...I'm Krishna Vamsi Rokkam", anchor=False)
    st.write("**🧠 AI Innovator | 👨🏻‍💻 Data Scientist  | 🤖 ML Engineer**")

    # Display social media icons    
    st.markdown("""
        <a href='https://www.linkedin.com/in/krishna-vamsi-rokkam/'>
        <img src="https://img.icons8.com/?size=100&id=13930&format=png&color=000000" alt="LinkedIn Profile" style="width:42px;height:42px;">
        </a>  |
        <a href="https://github.com/Krisshvamsi?tab=repositories/">
        <img src="https://icon-library.com/images/github-icon-white/github-icon-white-6.jpg" alt="GitHub Profile" style="width:42px;height:42px;">
        </a>
    """, unsafe_allow_html=True)
    st.write("\n")
    # Create columns for buttons within the second column
    button_col1, button_col2 = st.columns([0.2  , 0.8], gap="small")

    # Resume download button in the first button column
    with button_col1:
        with open("assets/CV_Krishna.pdf", "rb") as pdf_file:
            st.download_button(
                label="📄 Download Resume",
                data=pdf_file,
                file_name="Krishna's Resume.pdf",
                mime="application/pdf"
            )

    # Contact Me button in the second button column
    with button_col2:
        if st.button("✉️ Contact Me"):
            # Define the action for the "Contact Me" button
            show_contact_form()  # Define this function to show the contact form

st.write("\n")
st.write("\n")
st.write("\n")

# About Me Section
st.subheader("About Me 🌟", anchor=False)
st.markdown("""
    <hr style="margin-top: 1px; margin-bottom: 10px; border: 1px solid #ccc;">
""", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: justify;">
        <span style="color:#007acc;">In the dynamic world of <strong>AI</strong> and <strong>Data Science</strong></span>, I am a <strong>dot connector</strong>—passionate about unraveling complex challenges and crafting innovative solutions. My journey is driven by a deep curiosity to see beyond the obvious, connecting the dots between disparate data points 🔗 and transforming them into groundbreaking insights 💡.
        <br><br>  Having recently completed my <strong>Master’s degree in Applied Computer Science</strong> 🎓 from Concordia University in Montreal, I’ve cultivated a rich skill set that enables me to bridge gaps with technology. From engineering sophisticated <span style="color:#ff5722;"><strong>machine learning models</strong></span> 🤖 to deploying <span style="color:#ff5722;"><strong>AI solutions</strong></span> at scale 🌐, I leverage my expertise to make sense of the seemingly chaotic and turn it into clarity.
        <br><br>  My technical prowess spans across <span style="color:#4caf50;"><strong>Python software engineering</strong></span> 🐍, deep learning (including <strong>CNNs</strong> and <strong>Transformers</strong>) 🧠, <span style="color:#4caf50;"><strong>time series forecasting</strong></span> 📈, and <span style="color:#4caf50;"><strong>data augmentation</strong></span> 🔧. I've designed and deployed <span style="color:#2196f3;"><strong>CI/CD pipelines</strong></span> 🚀, orchestrated containers with <span style="color:#2196f3;"><strong>Docker</strong></span> 🐳 and <span style="color:#2196f3;"><strong>Kubernetes</strong></span> ⚙️, and built scalable <span style="color:#2196f3;"><strong>AI models</strong></span> using frameworks like <strong>PyTorch</strong>, <strong>TensorFlow</strong>, and <strong>Flask</strong>. My experience with <span style="color:#ff9800;"><strong>Power BI</strong></span> 📊 ensures that data-driven decisions are backed by clear, actionable insights.
        <br><br>  At the heart of my work is a relentless pursuit of <span style="color:#e91e63;"><strong>innovation</strong></span> 🚀, whether it’s enhancing <span style="color:#e91e63;"><strong>speech synthesis models</strong></span> 🗣️, developing <span style="color:#e91e63;"><strong>predictive tools</strong></span> 🔍, or fine-tuning <span style="color:#e91e63;"><strong>large language models</strong></span> for dialogue summarization 📝. I believe that the best solutions are those that connect ideas across domains, and I am always eager to engage with others, share my knowledge 📚, and explore new opportunities that push the boundaries of <strong>AI</strong> and <strong>Data Science</strong>.
        <br><br> If you’re looking for someone who thrives on solving complex problems 🧩 and connecting the dots in creative ways 🎨, let’s connect. Together, we can turn challenges into opportunities and drive forward the future of technology 🚀.
    </div>
""", unsafe_allow_html=True)

st.write("\n")
st.write("\n")

# Education
st.subheader("Education 📚", anchor=False)
st.markdown("""
    <hr style="margin-top: 1px; margin-bottom: 10px; border: 1px solid #ccc;">
""", unsafe_allow_html=True)
st.write("""
         - **Master of Applied Computer Science**  -  Concordia University, Montreal, Canada (September 2022 - April 2024)
         - **B. Tech in Information Technology**  -  Sreenidhi Institute of Science and Technology, Hyderabad, India (August 2018 - July 2022)
         """)

# Skills
st.subheader("Skills ⚙️🔧", anchor=False)
st.markdown("""
    <hr style="margin-top: 1px; margin-bottom: 10px; border: 1px solid #ccc;">
""", unsafe_allow_html=True)
st.write("""
- ***Programming languages***: C, C#, Python, R, Java
- ***Machine Learning***: PyTorch, Hugging Face, Deep learning, Transformers, LLM Fine Tuning, Keras, TensorFlow, Scikit-learn, SpeechBrain, MLFlow, OpenCV, CUDA
- ***Data Visualization & Analysis***: Tableau, Power BI, R-Studio, Matplotlib, Seaborn, Plotly, Pandas, Numpy
- ***Big Data & Databases***:  SQL, NoSQL (Couchbase, ElasticSearch)
- ***Cloud & DevOps***:  Azure AI, AWS (Sagemaker), Oracle cloud (OCI), Docker, Kubernetes, CI/CD, gRPC, Git, Flask, Streamlit  
""")


# Work Experience
st.subheader("Work Experience 💼", anchor=False)
st.markdown("""
    <hr style="margin-top: 1px; margin-bottom: 10px; border: 1px solid #ccc;">
""", unsafe_allow_html=True)
# Microsoft + Nuance
with st.expander("**Microsoft, Data Science Intern – Speech**, Montreal, Canada  (May - July, 2023)"):
    st.write("""
- Collaborated with the Professional Services Enterprise team to engineer a Python-based Regex Data Redaction Tool to safeguard sensitive PII and PCI data. Designed and implemented advanced regex patterns and integrated ML models (decision trees and SVMs), improving pattern recognition accuracy and redaction precision.
- Conducted comprehensive unit testing, ensuring robust performance and reducing manual redaction efforts by 50%.
- Co-developed an NLU Data Augmentation Tool using GPT-3/4 and Azure AI services to enhance training data generation. Improved model accuracy by 7% through benchmarking and hyperparameter tuning, and boosted project efficiency by 11% using custom evaluation metrics and detailed analysis with the Nuance Mix tool and Power BI.
""")


# Deloitte
with st.expander("**Deloitte, Data Science Intern**, Hyderabad, India (January - August, 2022)"):
    st.write("""
- Worked with the Deloitte Application Studio in Audit & Assurance to develop a Conversational AI bot using Azure LUIS and QnA Maker to handle complex audit user stories. Integrated with Excel data via API, reducing manual search time by 70%.
- Implemented Text-to-Speech capabilities for the bot using Azure TTS and Bot Framework services, significantly enhancing user engagement and accessibility by converting text-based responses to natural-sounding speech.
- Integrated Azure Cognitive Services using C#, including Azure Text Analytics API for sentiment analysis and key phrase extraction, and utilized the Direct Line Speech channel for seamless speech-to-text and text-to-speech interactions. This improved response accuracy and contextual understanding, increasing user satisfaction by 20%.
""")


# Technocolabs Softwares Inc
with st.expander("**Technocolabs Softwares Inc, Data Science Intern**, Remote (India) (May - August, 2021)"):
    st.write("""
- Executed data augmentation techniques such as SMOTE and temporal transformations (lag features, rolling statistics) to enhance a Time Series Forecasting project, resulting in a 15% improvement in model accuracy and F1 score. Used Pandas and Scikit-learn for preprocessing and augmentation.
- Developed predictive models for forecasting next-second stock price movements using Logistic Regression, SVM variants, and LSTM networks. LSTM outperformed other models with 15% higher accuracy and a 12% reduction in Mean Absolute Error (MAE).
- Utilized Grid Search and Random Search for hyperparameter optimization. Created interactive dashboards and visualizations with Tableau, improving insights communication and increasing decision-making efficiency by 30%.
             """)


# SmartBridge Pvt Ltd
with st.expander("**SmartBridge Pvt Ltd, ML Engineer Intern**, Hyderabad, India     (May - August, 2020)"):
    st.write("""
- Developed a Convolutional Neural Network (CNN) classification model for real-time skin disease detection, integrating OpenCV for efficient image preprocessing and feature extraction. Designed a custom PyTorch pipeline with image augmentation (rotation, scaling), normalization, and histogram equalization, enhancing model generalization and reducing overfitting.
- Engineered a CNN architecture using Max-Pooling, Flatten, and Conv2D layers with progressively larger filters to capture multi-scale features. Applied Batch Normalization and Dropout to enhance training stability and prevent overfitting.
- Leveraged transfer learning by fine-tuning the ResNet-50 model, reducing training time by 22% and improving accuracy by 4%. Deployed the model using Flask, with RESTful APIs for real-time image analysis and asynchronous processing to manage high user traffic, resulting in a scalable and responsive web application.
             """)
    

# Publications
st.subheader("Publications 📰", anchor=False)
st.markdown("""
    <hr style="margin-top: 1px; margin-bottom: 10px; border: 1px solid #ccc;">
""", unsafe_allow_html=True)
st.write("""
- **Co-authored book chapter**: *Post-COVID Impact on Skin Allergies* (Book Title: *[Data Science Applications of Post-COVID-19 Psychological Disorders](https://novapublishers.com/shop/data-science-applications-of-post-covid-19-psychological-disorders/)*)
- **An Intelligent TLDR Software for Summarization** – IJRASET ([DOI: 10.22214/ijraset.2022.44508](https://doi.org/10.22214/ijraset.2022.44508))
- **Predictive Analytics of BMI using CNN** – JMPAS ([DOI: 10.22270/jmpas.V10I6.1656](https://doi.org/10.22270/jmpas.V10I6.1656))
""")
    

# Function to create a project tile with justified text and buttons
def create_project_tile(column, image_path, title, description):
    with column:
        st.image(image_path, use_column_width=True)
        st.markdown(
            f"""
            <div style="background-color:#1e1e1e; padding:20px; border-radius:10px; color:white;">
                <h4 style="color:#66b2ff; margin-bottom: 10px;">{title}</h4>
                <p style="color:#e0e0e0; text-align: justify;">{description}</p>
            </div>
            <br>
            """, unsafe_allow_html=True
        )


# Projects Section
st.subheader("Projects 💻", anchor=False)
st.markdown("""
    <hr style="margin-top: 1px; margin-bottom: 10px; border: 1px solid #ccc;">
""", unsafe_allow_html=True)
st.write("\n")

# Creating a 3 by 3 grid for the project tiles
rows = [st.columns(3) for _ in range(3)]

# First Row
create_project_tile(rows[0][0], "assets/tts.png", "Transformer based TTS System", 
    """Implemented and optimized a Transformer-based  TTS model  using  the LJSpeech dataset  and SpeechBrain 
framework. Enhanced speech synthesis capabilities with scaled positional encodings, teacher forcing, and weighted 
loss functions, resulting in a Mel Error reduction to 8.27e-02 and a 10% decrease in Stop Error. Enhanced training efficiency and speech quality by utilizing dynamic batching, the Noam Scheduler, and Optimizer 
Initialization technique, achieving training speeds up to 2.5 times faster than Tacotron2, while improving long-range 
dependency handling for more natural speech synthesis """)

create_project_tile(rows[0][1], "assets/ds.jpg", "Dialogue Summarization: Fine Tuning LLM using Prompt Engineering and PEFT", 
    """Explored the FLAN-T5 model from Hugging Face for dialogue summarization. Utilized prompt engineering to refine 
summary  quality  and  experimented  with  zero/few-shot inference to enhance the model’s in-context  learning  and 
performance. Fine-tuned the model with the PEFT method – Low Rank Adaptation. Achieved a substantial reduction in model size 
while  maintaining  competitive  performance,  with  a  17.47%  improvement  in  ROUGE-1  and  8.73%  in  ROUGE-2 
scores over human baseline summaries """)

create_project_tile(rows[0][2], "assets/retail-and-consumer-goods-mobile.gif", "Automated Retail Product Classification using CNN", 
    """Developed  and  evaluated  CNN  architectures  (ResNet-18,  GoogleNet,  AlexNet)  for  grocery  product  classification. 
Tackled challenges such as data imbalance by implementing techniques like class weighting and oversampling, and 
mitigated vanishing gradients through careful initialization and normalization. Employed grid search for 
hyperparameter tuning, optimizing learning rates, batch sizes, and dropout rates to enhance model performance. Applied transfer learning by fine-tuning the ResNet-18 model with pre-trained weights on ImageNet. Achieved an 8% 
improvement in classification accuracy and performed bias analysis to evaluate model performance across different 
product categories, uncovering insights into model weaknesses and areas for further refinement""")

# Second Row
create_project_tile(rows[1][0], "assets/wine.jpg", "Modeling Wine Quality using Ensemble Modeling Approach", 
    """Applied ensemble methods, including Random Forest, Gradient Boosting, AdaBoost, and XGBoost, to predict wine 
quality  based  on  chemical properties and  expert  evaluations.  Utilized  bagging and  boosting techniques to  enhance 
model robustness and accuracy. Achieved 89% prediction accuracy, with XGBoost outperforming other models. This approach provided a 
comprehensive and reliable assessment of wine quality, improving evaluation accuracy and analytical insights""")

create_project_tile(rows[1][1], "assets/digit.gif", "Multi-task Modeling on handwritten digits using Keras", 
   """Performed simultaneous tasks on grayscale digits: predicting digit value and color. Designed a data generator function that
generates red, green colour images using the greyscale MNIST images dataset from Keras. Developed a Resnet-style architecture with skip connections for a multi-tasking neural network model. Achieved a
remarkable 98% accuracy for digit and color recognition using interconnected neural networks""")

create_project_tile(rows[1][2], "assets/skin.png", "Skin Disease Identification using Image Analysis", 
    """Developed a Convolutional Neural Network (CNN) classification model for real-time skin disease detection, integrating 
OpenCV  for  efficient  image  preprocessing  and  feature  extraction.  Designed  a  custom  PyTorch  pipeline  with  image 
augmentation  (rotation,  scaling),  normalization,  and  histogram  equalization,  enhancing  model  generalization  and 
reducing overfitting. Applied  Batch  Normalization  and  Dropout  to  enhance  training  stability  and  prevent 
overfitting. Leveraged  transfer  learning  by  fine-tuning  the  ResNet-50  model,  reducing  training  time  by  22%  and  improving 
accuracy by 4%. Deployed the model using Flask, with RESTful APIs for real-time image analysis and asynchronous 
processing to manage high user traffic, resulting in a scalable and responsive web application""")

# # Third Row
# create_project_tile(rows[2][0], "assets/menu.gif", "Projectd Title", 
#     "Project description goes here. This is where you briefly describe the project and your contributions.")

# create_project_tile(rows[2][1], "assets/menu.gif", "Projecst Title", 
#     "Project description goes here. This is where you briefly describe the project and your contributions.")

# create_project_tile(rows[2][2], "assets/menu.gif", "Projesvct Title", 
#     "Project description goes here. This is where you briefly describe the project and your contributions.")

st.write("\n")

# Certifications
st.subheader("Certifications 📜", anchor=False)
st.markdown("""
    <hr style="margin-top: 1px; margin-bottom: 10px; border: 1px solid #ccc;">
""", unsafe_allow_html=True)
st.write("""
- **Oracle Cloud Infrastructure Generative AI Professional certification** – Oracle Cloud
- **Machine Learning certification** – Stanford University
- **AWS Academy - Cloud Foundations course** – Amazon
- **Programming Data Structures and Algorithms using Python** – IIT Madras
""")


# Achievements
st.subheader("Achievements 🏅", anchor=False)
st.markdown("""
    <hr style="margin-top: 1px; margin-bottom: 10px; border: 1px solid #ccc;">
""", unsafe_allow_html=True)
st.write("""
- Full-Time Post Matric Scholarship from the State Government of Telangana for the entire bachelor’s degree
- Silver Award at Ennovate - The International Innovation Show, Poland, 2021
- ELITE grade in 'Data Structures and Algorithms using Python' course from IIT Madras
- Bronze Award in Global Assessment of Information Technology, 2021
""")


# Extra-Curricular Activities
st.subheader("Activities and Societies 🌐🤝", anchor=False)
st.markdown("""
    <hr style="margin-top: 1px; margin-bottom: 10px; border: 1px solid #ccc;">
""", unsafe_allow_html=True)
# Volunteer Team Leader
with st.expander("**Volunteer Team Leader, ConUHacks VIII, Concordia University**,  (Jan 2024)"):
    st.write("""
- Led a team of volunteers, coordinating activities at Quebec’s largest hackathon. Demonstrated strong leadership, task assignment, and communication skills.
- Fostered a collaborative environment, addressed issues promptly, and ensured a positive experience for participants, sponsors, and executives, contributing to the event's success.
""")

# The Techvision Club
with st.expander("**The Techvision Club**,  (Aug 2020 – Sep 2022)"):
    st.write("""
- **Board Member & Executive**: Led the Focus on Research (FOR) initiative, mentoring students in research projects and facilitating publications.
- **Programming Tutor**: Conducted hands-on training sessions in C, Python, and R for over 120 students.
- **Event Organizer**: Developed the club's inaugural website and organized technical events including workshops, hackathons, and quizzes. Hosted webinars on AI, Data Science, and Web Development with industry experts.
""")

# AIESEC in Hyderabad
with st.expander("**AIESEC in Hyderabad**,  (Aug 2020 – Jan 2021)"):
    st.write("""
- Contributed to the Outgoing Global Talent Department, specializing in content creation, market research, and partnership development for international internship exchanges.
- Collaborated with a global network of young leaders to promote cross-cultural understanding and facilitate international internships.
""")