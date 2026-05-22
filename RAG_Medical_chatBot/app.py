import streamlit as st
from setting import DATA_PATH
from primary.load import load_data
from models.base import BaseEmbeddingModel
from rag.retriever import Retriever
from rag.pipeline import run_rag
from llm.generator import LLMGenerator


# PAGE CONFIG
st.set_page_config(
    page_title="Diabetes Medical Chatbot",
    page_icon="🩺",
    layout="wide"
)


# TITLE
st.title("🩺 Diabetes Medical Chatbot")
st.markdown(
    "RAG + TinyLlama + LoRA Fine-Tuning"
)


# LOAD DATA
@st.cache_resource
def load_chatbot():
    # load dataset
    df, texts = load_data(DATA_PATH)
    # embedding model
    embedding_model = BaseEmbeddingModel(
        "all-MiniLM-L6-v2"
    )
    # embeddings
    embeddings = embedding_model.encode(texts)
    # retriever
    retriever = Retriever(embeddings)
    # llm
    llm = LLMGenerator(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    return (
        df,
        embedding_model,
        retriever,
        llm
    )


# LOAD COMPONENTS
with st.spinner("Loading chatbot..."):

    (
        df,
        embedding_model,
        retriever,
        llm
    ) = load_chatbot()

st.success("Chatbot loaded successfully!")

# SESSION MEMORY
if "messages" not in st.session_state:
    st.session_state.messages = []


# DISPLAY CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# USER INPUT
prompt = st.chat_input(
    "Ask about diabetes..."
)


# CHATBOT RESPONSE
if prompt:
    # show user message
    st.chat_message("user").markdown(prompt)
    # save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })


    # RAG RETRIEVAL
    results = run_rag(
        prompt,
        embedding_model,
        retriever,
        df,
        k=3
    )
    # context
    context = "\n\n".join(
        results["context"].tolist()
    )


    # GENERATE RESPONSE
    with st.spinner("Generating response..."):
        response = llm.generate(
            prompt,
            context
        )


    # SHOW BOT RESPONSE
    with st.chat_message("assistant"):
        st.markdown(response)
        # optional context viewer
        with st.expander("Retrieved Context"):
            st.write(context)

    # save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })