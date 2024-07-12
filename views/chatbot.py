import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings


openai.api_key = st.secrets["openai_key"]
st.title("AMA about Krishna, powered by LlamaIndex 💬🦙")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Streamlit's open-source Python library!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""You are an expert on 
        the Streamlit Python library and your 
        job is to answer technical questions. 
        Assume that all questions are related 
        to the Streamlit Python library. Keep 
        your answers technical and based on 
        facts – do not hallucinate features.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)







# import streamlit as st
# import requests
# import streamlit.components.v1 as components
# from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext
# from llama_index.llm_predictor import  LLMPredictor
# from PIL import Image
# import openai
# from langchain.chat_models import ChatOpenAI

# st.set_page_config(page_title='Template' ,layout="wide",page_icon='👧🏻')



# # -----------------  chatbot  ----------------- #
# # Set up the OpenAI key
# openai_api_key = st.sidebar.text_input('Enter your OpenAI API Key and hit Enter', type="password")
# openai.api_key = (openai_api_key)

# # load the file
# documents = SimpleDirectoryReader(input_files=["bio.txt"]).load_data()

# pronoun = "He | Him"
# name = "Krishna Vamsi Rokkam"
# def ask_bot(input_text):
#     # define LLM
#     llm = ChatOpenAI(
#         model_name="gpt-3.5-turbo",
#         temperature=0,
#         openai_api_key=openai.api_key,
#     )
#     llm_predictor = LLMPredictor(llm=llm)
#     service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    
#     # load index
#     index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)    
    
#     # query LlamaIndex and GPT-3.5 for the AI's response
#     PROMPT_QUESTION = f"""You are Buddy, an AI assistant dedicated to assisting {name} in her job search by providing recruiters with relevant and concise information. 
#     If you do not know the answer, politely admit it and let recruiters know how to contact {name} to get more information directly from {pronoun}. 
#     Don't put "Buddy" or a breakline in the front of your answer.
#     Human: {input}
#     """
    
#     output = index.as_query_engine().query(PROMPT_QUESTION.format(input=input_text))
#     print(f"output: {output}")
#     return output.response

# # get the user's input by calling the get_text function
# def get_text():
#     input_text = st.text_input("After providing OpenAI API Key on the sidebar, you can send your questions and hit Enter to know more about me from my AI agent, Buddy!", key="input")
#     return input_text

# #st.markdown("Chat With Me Now")
# user_input = get_text()

# if user_input:
#   #text = st.text_area('Enter your questions')
#   if not openai_api_key.startswith('sk-'):
#     st.warning('⚠️Please enter your OpenAI API key on the sidebar.', icon='⚠')
#   if openai_api_key.startswith('sk-'):
#     st.info(ask_bot(user_input))