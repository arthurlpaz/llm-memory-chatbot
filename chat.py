import streamlit as st
import os
import tempfile

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ENV
load_dotenv()

# STREAMLIT SETUP
st.set_page_config(page_title="LLM Memory Chatbot", page_icon="🤖")
st.title("🤖 I'm your virtual intelligence assistant")

# MODEL SELECTION
model_class = "hf_hub"  # Options: "hf_hub", "openai", "ollama"


# MODELS
def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
    llm = HuggingFaceEndpoint(
        repo_id=model,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=temperature,
        max_new_tokens=512,
        streaming=True,
    )

    chat_llm = ChatHuggingFace(llm=llm)
    return chat_llm


def model_openai(model="gpt-4o-mini", temperature=0.1):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=True,
    )


def model_ollama(model="phi3", temperature=0.1):
    return ChatOllama(
        model=model,
        temperature=temperature,
        streaming=True,
    )


def model_groq(model="llama3-70b-8192", temperature=0.1):
    llm = ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    return llm


# SESSION STATE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your virtual assistant 🤖. How can I help you?")
    ]

# CHAT HISTORY RENDER
for message in st.session_state.chat_history:
    role = "ai" if isinstance(message, AIMessage) else "human"
    with st.chat_message(role):
        st.markdown(message.content)

# Indexation and recoveration
def setup_retriever(uploads):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embedding
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Storage
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("vectorstore/db_faiis")

    # Setup retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4}
    )

    return retriever


# SETUP CHAIN
def setup_rag_chain(model_class, retriever):
    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()
    elif model_class == "groq":
        llm = model_groq()
    else:
        raise ValueError("Invalid model_class")

    # Prompt definition
    if model_class.startswith("hf"):
        # token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" # llama 3
        token_s, token_e = "<|system|>", "<|end|><|assistant|>"  # phi3
    else:
        token_s, token_e = "", ""

    context_q_system_prompt = """
    Given the following chat history and the follow-up question which might reference context 
    in the chat history, formulate a standalone question which can be understood without the 
    chat history. Do NOT answer the question, just reformulate it if needed and otherwise 
    return it as is.
    """

    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = "Question: {input}" + token_e
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", context_q_user_prompt),
        ]
    )


    # Chain to contextualize 
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=context_q_prompt
    )

    qa_prompt_template = """Você é um assistente virtual prestativo e está respondendo perguntas gerais.
    Use os seguintes pedaços de contexto recuperado para responder à pergunta.
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain,
    )

    return rag_chain

# File uploader in sidebar
uploads = st.sidebar.file_uploader(
    label="Upload a file to add context",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Please, upload at least one file to provide context for the chatbot")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm your virtual assistant! How can I help you?"),
    ]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

user_query = st.chat_input("Type here your message...")

if user_query is not None and user_query != "" and uploads is not None:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("human"):
        st.markdown(user_query)

    with st.chat_message("ai"):

        if st.session_state.docs_list != uploads:
            print(uploads)
            st.session_state.docs_list = uploads
            st.session_state.retriever = setup_retriever(uploads)

        rag_chain = setup_rag_chain(model_class, st.session_state.retriever)

        result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})

        resp = result['answer']
        st.write(resp)

        # mostrar a fonte
        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', 'Página não especificada')

            ref = f":link: Fonte {idx}: *{file} - p. {page}*"
            print(ref)
            with st.popover(ref):
                st.caption(doc.page_content)

    st.session_state.chat_history.append(AIMessage(content=resp))
