import streamlit as st
import numpy as np
import time
import os


from setting import DATA_PATH

from primary.load import load_data

from rag.retriever import Retriever

from rag.langchain_pipeline import (
    build_rag_chain
)

from llm.generator import (
    LLMGenerator
)

from models.albert import (
    AlbertModel
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title=(
        "DiabeteBot | AI Medical Assistant"
    ),

    page_icon="🩺",

    layout="wide",

    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD CHATBOT
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_chatbot():

    print("=" * 50)
    print("START LOADING CHATBOT")
    print("=" * 50)

    start = time.time()

    print(
        "Current Working Directory:"
    )

    print(
        os.getcwd()
    )

    # ========================================================
    # STEP 1 - DATASET
    # ========================================================

    print(
        "STEP 1 - Loading dataset..."
    )

    df, texts = load_data(
        DATA_PATH
    )

    print(
        f"Dataset loaded "
        f"({len(df)} rows) "
        f"in {time.time()-start:.2f}s"
    )

    # ========================================================
    # STEP 2 - ALBERT
    # ========================================================

    t = time.time()

    print(
        "STEP 2 - Loading ALBERT embedding model..."
    )

    embedding_model = AlbertModel()

    print(
        f"Embedding model loaded "
        f"in {time.time()-t:.2f}s"
    )

    # ========================================================
    # STEP 3 - EMBEDDINGS
    # ========================================================

    t = time.time()

    print(
        "STEP 3 - Loading embeddings.npy..."
    )

    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    EMBEDDING_PATH = os.path.join(
        BASE_DIR,
        "embedding",
        "embeddings.npy"
    )

    print(
        "BASE_DIR:",
        BASE_DIR
    )

    print(
        "EMBEDDING_PATH:",
        EMBEDDING_PATH
    )

    print(
        "FILE EXISTS:",
        os.path.exists(
            EMBEDDING_PATH
        )
    )

    embeddings = np.load(
        EMBEDDING_PATH
    )

    print(
        f"Embeddings loaded "
        f"Shape={embeddings.shape} "
        f"in {time.time()-t:.2f}s"
    )

    # ========================================================
    # STEP 4 - FAISS
    # ========================================================

    t = time.time()

    print(
        "STEP 4 - Building FAISS retriever..."
    )

    faiss_retriever = Retriever(
        embeddings
    )

    print(
        f"FAISS retriever ready "
        f"in {time.time()-t:.2f}s"
    )

    # ========================================================
    # STEP 5 - TINYLLAMA + LORA
    # ========================================================

    t = time.time()

    print(
        "STEP 5 - Loading TinyLlama + LoRA..."
    )

    llm = LLMGenerator(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    print(
        f"LLM loaded "
        f"in {time.time()-t:.2f}s"
    )

    # ========================================================
    # STEP 6 - LANGCHAIN RAG CHAIN
    # ========================================================

    t = time.time()

    print(
        "STEP 6 - Building LangChain RAG pipeline..."
    )

    rag_chain = build_rag_chain(

        embedding_model=embedding_model,

        faiss_retriever=faiss_retriever,

        dataframe=df,

        llm_generator=llm
    )

    print(
        f"LangChain RAG pipeline ready "
        f"in {time.time()-t:.2f}s"
    )

    # ========================================================
    # READY
    # ========================================================

    print(
        "STEP 7 - Chatbot ready!"
    )

    print(
        f"TOTAL LOAD TIME: "
        f"{time.time()-start:.2f}s"
    )

    return (
        df,
        rag_chain
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🩺 DiabetesBot"
    )

    st.caption(
        "AI Medical Assistant • "
        "RAG + LoRA + LangChain"
    )

    st.divider()

    with st.status(
        "Initializing System Models...",
        expanded=True
    ) as status:

        df, rag_chain = load_chatbot()

        status.update(

            label=(
                "All Systems Operational!"
            ),

            state="complete",

            expanded=False
        )

    st.divider()

    st.markdown(
        "### 💡 How To Use"
    )

    st.markdown(
        "- Type a question about **Diabetes**.\n"
        "- The system retrieves relevant documents.\n"
        "- FAISS retrieves the top 5 candidates.\n"
        "- CrossEncoder reranks them to the top 3.\n"
        "- TinyLlama generates an answer using the context.\n"
        "- Open the references to view the retrieved context."
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat History",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "Diabetes Medical Q&A"
)

st.markdown(
    "Hello! I am an AI medical assistant "
    "powered by a *fine-tuned* language model."
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and "context" in message
        ):

            with st.expander(
                "📚 View Document References"
            ):

                st.info(
                    message["context"]
                )


# ============================================================
# USER INPUT
# ============================================================

if prompt := st.chat_input(
    "Type your question here..."
):

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.chat_message(
        "user"
    ).markdown(
        prompt
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching, reranking, and generating..."
        ):

            try:

                # ============================================
                # LANGCHAIN CHAIN
                # ============================================

                result = rag_chain.invoke(
                    {
                        "question": prompt
                    }
                )

                # ============================================
                # GET ANSWER
                # ============================================

                response = result[
                    "answer"
                ]

                # ============================================
                # GET FINAL TOP-3 CONTEXT
                # ============================================

                context = result[
                    "context"
                ]

                # ============================================
                # DISPLAY ANSWER
                # ============================================

                st.markdown(
                    response
                )

                # ============================================
                # DISPLAY RETRIEVED DOCUMENTS
                # ============================================

                with st.expander(
                    "📚 Retrieved Documents"
                ):

                    st.info(
                        context
                    )

                # ============================================
                # SAVE CHAT
                # ============================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "context": context
                    }
                )

            except Exception as e:

                st.error(
                    "⚠️ An error occurred "
                    f"in the system: {str(e)}"
                )