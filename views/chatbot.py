from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
import os

if "history" not in st.session_state:
    st.session_state.history = []

load_dotenv()

model_type= 'gemini'

st.header("Welcome to AMA about Krishna, developed using LangChain, LLMs and RAG", anchor=False)
st.write("""
        Please let me know your questions about Krishna
         """)

# Initializing Gemini
if(model_type == "ollama"):
    model = Ollama(
                    model="llama3:latest",  # Provide your ollama model name here
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler])
                )
    
elif(model_type == "gemini"):
    model = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.1, 
                                convert_system_message_to_human=True
                            )
    
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Vector Database
persist_directory = "./db/gemini/" # Persist directory path
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if not os.path.exists(persist_directory):
    with st.spinner('🚀 Starting the bot...'):
        # Data Pre-processing
        pdf_loader = DirectoryLoader("./data/", glob="./*.pdf", loader_cls=PyPDFLoader)
        text_loader = DirectoryLoader("./data/", glob="./*.txt", loader_cls=TextLoader)
        
        pdf_documents = pdf_loader.load()
        text_documents = text_loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        
        pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
        text_context = "\n\n".join(str(p.page_content) for p in text_documents)

        pdfs = splitter.split_text(pdf_context)
        texts = splitter.split_text(text_context)

        data = pdfs + texts

        print("Data Processing Complete")

        vectordb = Chroma.from_texts(data, embeddings, persist_directory=persist_directory)
        vectordb.persist()

        print("Vector DB Creating Complete\n")

elif os.path.exists(persist_directory):
    vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embeddings)
    
    print("Vector DB Loaded\n")

# Quering Model
query_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectordb.as_retriever()
)

for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])


prompt = st.chat_input("Say something")
if prompt:
    st.session_state.history.append({
        'role':'user',
        'content':prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('💡Thinking'):
        response = query_chain({"query": prompt})

        st.session_state.history.append({
            'role' : 'Assistant',
            'content' : response['result']
        })

        with st.chat_message("Assistant"):
            st.markdown(response['result'])


# import streamlit as st
# import openai
# from langchain.chat_models import ChatOpenAI
# from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, LLMPredictor

# st.set_page_config(page_title='Template', layout="wide", page_icon='👧🏻')

# # -----------------  chatbot  ----------------- #
# # Set up the OpenAI key
# openai_api_key = st.sidebar.text_input('Enter your OpenAI API Key and hit Enter', type="password")
# openai.api_key = openai_api_key

# # Load the file
# documents = SimpleDirectoryReader(input_files=["bio.txt"]).load_data()

# pronoun = "He | Him"
# name = "Krishna Vamsi Rokkam"

# def ask_bot(input_text):
#     # Define LLM
#     llm = ChatOpenAI(
#         model_name="gpt-3.5-turbo",
#         temperature=0,
#         openai_api_key=openai.api_key,
#     )
#     llm_predictor = LLMPredictor(llm=llm)
#     service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

#     # Load index
#     index = VectorStoreIndex.from_documents(documents, service_context=service_context)

#     # Query LlamaIndex and GPT-3.5 for the AI's response
#     PROMPT_QUESTION = f"""You are Buddy, an AI assistant dedicated to assisting {name} in his job search by providing recruiters with relevant and concise information. 
#     If you do not know the answer, politely admit it and let recruiters know how to contact {name} to get more information directly from him. 
#     Don't put "Buddy" or a breakline in the front of your answer.
#     Human: {input_text}
#     """

#     output = index.as_query_engine().query(PROMPT_QUESTION.format(input=input_text))
#     print(f"output: {output}")
#     return output.response

# # Get the user's input
# def get_text():
#     input_text = st.text_input("After providing OpenAI API Key on the sidebar, you can send your questions and hit Enter to know more about me from my AI agent, Buddy!", key="input")
#     return input_text

# user_input = get_text()

# if user_input:
#     if not openai_api_key.startswith('sk-'):
#         st.warning('⚠️Please enter your OpenAI API key on the sidebar.', icon='⚠')
#     if openai_api_key.startswith('sk-'):
#         st.info(ask_bot(user_input))
