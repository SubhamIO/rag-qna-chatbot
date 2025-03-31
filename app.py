import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import bs4
from langchain.chains import create_history_aware_retriever
from tqdm.autonotebook import tqdm as notebook_tqdm
import sys
from langchain_community.document_loaders import PyPDFLoader


# import chromadb
# chromadb.api.client.SharedSystemClient.clear_system_cache()
from langchain_community.vectorstores import FAISS

load_dotenv()

LANGCHAIN_API_KEY = st.secrets['LANGCHAIN_API_KEY']
HF_TOKEN = st.secrets['HF_TOKEN']

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot - Ask Subham!"
# groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"]=HF_TOKEN

def main():
    ## #Title of the app
    st.title("Enhanced RAG Q&A Chatbot With ChatGroq and HuggingFace")
    ## Input the Groq API Key
    api_key=st.text_input("Enter your Groq API key:",type="password")
    groq_api_key = api_key
    if api_key:
        
        ## Select the OpenAI model
        llm=st.sidebar.selectbox("Select Open Source model",["Llama3-8b-8192","Gemma2-9b-It"])
        embedding=st.sidebar.selectbox("Select Embedding model",["all-MiniLM-L6-v2"])

        session_id=st.sidebar.text_input("Enter a Session Id:")

        ## Adjust response parameter
        temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
        max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

        uploaded_files=st.sidebar.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)

        ## Main interface for user input
        st.write("Go ahead and ask any question...")
        question=st.text_input("You:")


        embeddings=HuggingFaceEmbeddings(model_name=embedding)
        model = ChatGroq(groq_api_key=groq_api_key,model=llm,temperature=temperature,max_tokens=max_tokens) 

        if 'store' not in st.session_state:
                st.session_state.store={}

        ## Process uploaded  PDF's
        if uploaded_files:
            documents=[]
            for uploaded_file in uploaded_files:
                temppdf=f"./temp.pdf"
                with open(temppdf,"wb") as file:
                    file.write(uploaded_file.getvalue())
                    file_name=uploaded_file.name

                loader=PyPDFLoader(temppdf)
                docs=loader.load()
                documents.extend(docs)

            ## Split the document into chunks
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
            splits=text_splitter.split_documents(documents)

            ## Store the embedded chunks to the vector DB
            vectorstore=FAISS.from_documents(documents=splits,embedding=embeddings)
            retriever=vectorstore.as_retriever()

            ## this function will create a session id 
            ## this session id will be needed to distinguish different chats history
            def get_session_history(session:str)->BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id]=ChatMessageHistory()
                return st.session_state.store[session_id]


            ## contextual prompt - this will convert the human question to more understandable question text based on the context 
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            ## upgrading the retriever to remember history
            history_aware_retriever=create_history_aware_retriever(model,retriever,contextualize_q_prompt)

            ## initialise chatprompttemplate 
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Please provide the most accurate response based on the question"
                "\n\n"
                "{context}"
            )

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_document_chain=create_stuff_documents_chain(model,qa_prompt)
            chain=create_retrieval_chain(history_aware_retriever,question_answer_document_chain)


            ## initialise chat memory
            ## Let's now wrap this more complicated chain in a Message History class. 
            ## This time, because there are multiple keys in the input, we need to specify the correct key to use to save the chat history.
            with_message_history = RunnableWithMessageHistory(
                    chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )

            ## create new session
            config = {"configurable": {"session_id": session_id}}

            ## chat with LLM
            if question:
                session_history=get_session_history(session_id)
                answer=with_message_history.invoke(
                        {'input': question},
                        config=config
                    )["answer"]
                
                st.write(st.session_state.store)
                st.write("Assistant:", answer)
                st.write("Chat History:", session_history.messages)

            else:
                st.write("Please provide the user input")  

    else:
        st.warning("Please enter the Groq API Key")


if __name__=='__main__':
    main()
