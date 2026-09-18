import streamlit as st
import numpy as np
import time
import os

from setting import DATA_PATH
from primary.load import load_data
from rag.retriever import Retriever
from rag.pipeline import run_rag
from llm.generator import LLMGenerator
from models.albert import AlbertModel


# PAGE CONFIGURATION
st.set_page_config(
    page_title="DiabeteBot | AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD CHATBOT

@st.cache_resource(show_spinner=False)
def load_chatbot():

    print("=" * 50)
    print("START LOADING CHATBOT")
    print("=" * 50)

    start = time.time()

    print("Current Working Directory:")
    print(os.getcwd())


    print("STEP 1 - Loading dataset...")
    df, texts = load_data(DATA_PATH)
    print(
        f"Dataset loaded ({len(df)} rows) "
        f"in {time.time()-start:.2f}s"
    )

    t = time.time()
    print("STEP 2 - Loading ALBERT embedding model...")
    embedding_model = AlbertModel()
    print(
        f"Embedding model loaded "
        f"in {time.time()-t:.2f}s"
    )

    t = time.time()
    print("STEP 3 - Loading embeddings.npy...")
    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )
    EMBEDDING_PATH = os.path.join(
        BASE_DIR,
        "embedding",
        "embeddings.npy"
    )
    print("BASE_DIR:", BASE_DIR)
    print("EMBEDDING_PATH:", EMBEDDING_PATH)
    print("FILE EXISTS:", os.path.exists(EMBEDDING_PATH))

    embedding_dir = os.path.join(
        BASE_DIR,
        "embedding"
    )
    if os.path.exists(embedding_dir):
        print("Embedding folder contents:")
        print(os.listdir(embedding_dir))
    else:
        print("Embedding folder not found!")
    embeddings = np.load(
        EMBEDDING_PATH
    )
    print(
        f"Embeddings loaded "
        f"Shape={embeddings.shape} "
        f"in {time.time()-t:.2f}s"
    )

    t = time.time()
    print("STEP 4 - Building retriever...")
    retriever = Retriever(embeddings)
    print(
        f"Retriever ready "
        f"in {time.time()-t:.2f}s"
    )

    t = time.time()
    print("STEP 5 - Loading TinyLlama + LoRA...")
    llm = LLMGenerator("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(
        f"LLM loaded "
        f"in {time.time()-t:.2f}s"
    )

    print("STEP 6 - Chatbot ready!")
    print(
        f"TOTAL LOAD TIME: "
        f"{time.time()-start:.2f}s"
    )
    return (
        df,
        embedding_model,
        retriever,
        llm
    )

with st.sidebar:
    st.title("🩺 DiabetesBot")
    st.caption("AI Medical Assistant • RAG + LoRA")
    st.divider()
    with st.status(
        "Initializing System Models...",
        expanded=True
    ) as status:
        df, embedding_model, retriever, llm = load_chatbot()
        status.update(
            label="All Systems Operational!",
            state="complete",
            expanded=False
        )
    st.divider()
    st.markdown("### 💡 How To Use")
    st.markdown(
        "- Type a question about **Diabetes**.\n"
        "- The system will retrieve relevant documents.\n"
        "- The answer will be generated based on the retrieved context.\n"
        "- Open the references to view the source context."
    )

    st.divider()
    if st.button(
        "🗑️ Clear Chat Hiatory",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

st.title("Diabetes Medical Q&A")
st.markdown(
    "Hello! I am an AI medical assistant powered by a *fine-tuned* language model."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(message["content"])
        if (message["role"] == "assistant"and "context" in message):
            with st.expander("📚 View Document References"):
                st.info(message["context"])


if prompt := st.chat_input("Type your question here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    with st.chat_message("assistant"):
        with st.spinner(
            "Searching for relevant information and generating an answer..."
        ):
            try:
                results = run_rag(
                    prompt,
                    embedding_model,
                    retriever,
                    df,
                    k=3
                )
                context = "\n\n---\n\n".join(results["contexts"])
                response = llm.generate(prompt,context)
                st.markdown(response)
                with st.expander("📚 Retrieved Documents"):
                    st.info(context)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "context": context
                    }
                )

            except Exception as e:
                st.error(
                    f"⚠️ An error occurred in the system.: {str(e)}"
                )