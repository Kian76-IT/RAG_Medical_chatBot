import streamlit as st

from setting import DATA_PATH

from primary.load import load_data

from models.albert import AlbertModel

from rag.retriever import Retriever
from rag.pipeline import run_rag

from llm.generator import LLMGenerator


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Diabetes Medical Chatbot",
    page_icon="🩺",
    layout="wide"
)


# =====================================
# TITLE
# =====================================

st.title("🩺 Diabetes Medical Chatbot")

st.markdown("""
### Retrieval-Augmented Generation (RAG)

**Retriever:** ALBERT

**Generator:** TinyLlama + LoRA Fine-Tuning
""")


# =====================================
# LOAD COMPONENTS
# =====================================

@st.cache_resource
def load_chatbot():

    # DATASET
    df, texts = load_data(DATA_PATH)

    # ALBERT EMBEDDING MODEL
    embedding_model = AlbertModel()

    # DOCUMENT EMBEDDINGS
    embeddings = embedding_model.encode(
        texts
    )

    # RETRIEVER
    retriever = Retriever(
        embeddings
    )

    # LLM + LoRA
    llm = LLMGenerator(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    return (
        df,
        embedding_model,
        retriever,
        llm
    )


# =====================================
# LOAD CHATBOT
# =====================================

with st.spinner(
    "Loading chatbot..."
):

    (
        df,
        embedding_model,
        retriever,
        llm
    ) = load_chatbot()

st.success(
    "Chatbot loaded successfully!"
)


# =====================================
# SESSION STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================
# DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


# =====================================
# USER INPUT
# =====================================

prompt = st.chat_input(
    "Ask a diabetes-related question..."
)


# =====================================
# CHATBOT RESPONSE
# =====================================

if prompt:

    # USER MESSAGE
    st.chat_message(
        "user"
    ).markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # =====================================
    # RAG RETRIEVAL
    # =====================================

    results = run_rag(
        query=prompt,
        embedding_model=embedding_model,
        retriever=retriever,
        df=df,
        k=3
    )

    # COMBINE CONTEXT
    context = "\n\n".join(
        results["contexts"]
    )

    # =====================================
    # GENERATE RESPONSE
    # =====================================

    with st.spinner(
        "Generating response..."
    ):

        response = llm.generate(
            prompt,
            context
        )

    # =====================================
    # SHOW RESPONSE
    # =====================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            response
        )

        with st.expander(
            "Retrieved Context"
        ):
            st.write(
                context
            )

    # SAVE CHAT HISTORY

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )