import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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


# RESPONSE PIPELINE
def model_response(user_query, chat_history, model_class):

    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()
    else:
        raise ValueError("Invalid model_class")

    system_prompt = (
        "Você é um assistente prestativo e responde perguntas gerais. "
        "You're a helpful assistant and answers general questions."
        "Always answer in {language}."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.stream(
        {"chat_history": chat_history, "input": user_query, "language": "english"}
    )


# SESSION STATE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your virtual assistant 🤖. How can I help you?")
    ]

# CHAT HISTORY RENDER
for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.markdown(message.content)

# USER INPUT
user_query = st.chat_input("Type here your message...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(
            model_response(user_query, st.session_state.chat_history, model_class)
        )

    st.session_state.chat_history.append(AIMessage(content=response))
