from operator import itemgetter

from sentence_transformers import CrossEncoder

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)
from langchain_core.output_parsers import StrOutputParser

from rag.retriever import (
    AlbertFaissRetriever,
    Retriever
)


# ============================================================
# CONFIGURATION
# ============================================================

FAISS_TOP_K = 5
RERANK_TOP_K = 3


# ============================================================
# PROMPT
# ============================================================

MEDICAL_PROMPT = PromptTemplate.from_template(
    """
<|system|>

You are a medical question answering assistant
specialized in diabetes.

Use ONLY the information provided in the context.

Rules:
1. Answer the question directly.
2. Do not invent medical information.
3. Do not use information outside the context.
4. Do not add greetings.
5. Do not add motivational statements.
6. Do not add unnecessary conclusions.
7. Do not repeat the context.
8. Do not repeat the question.
9. If the answer cannot be found in the context,
   say exactly:

"I don't know based on the provided context."

<|context|>

{context}

<|user|>

{question}

<|assistant|>
"""
)


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_documents(
    documents
):

    if not documents:

        return (
            "No relevant context was retrieved."
        )

    formatted = []

    for i, document in enumerate(
        documents,
        start=1
    ):

        formatted.append(
            f"[Document {i}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(
        formatted
    )


# ============================================================
# RERANK DOCUMENTS
# ============================================================

class DocumentReranker:

    def __init__(
        self,
        model_name=(
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        ),
        top_k=RERANK_TOP_K
    ):

        print(
            "Loading CrossEncoder reranker..."
        )

        self.model = CrossEncoder(
            model_name
        )

        self.top_k = top_k

        print(
            "CrossEncoder loaded!"
        )

    def rerank(
        self,
        query,
        documents
    ):

        if not documents:

            return []

        # ----------------------------------------------------
        # CREATE QUERY-DOCUMENT PAIRS
        # ----------------------------------------------------

        pairs = [
            (
                query,
                document.page_content
            )
            for document in documents
        ]

        # ----------------------------------------------------
        # CROSS ENCODER SCORE
        # ----------------------------------------------------

        scores = self.model.predict(
            pairs
        )

        # ----------------------------------------------------
        # SORT BY SCORE
        # ----------------------------------------------------

        ranked = sorted(
            zip(
                documents,
                scores
            ),
            key=lambda x: float(x[1]),
            reverse=True
        )

        # ----------------------------------------------------
        # TOP N
        # ----------------------------------------------------

        selected = []

        for rank, (
            document,
            score
        ) in enumerate(
            ranked[:self.top_k],
            start=1
        ):

            document.metadata[
                "rerank_score"
            ] = float(score)

            document.metadata[
                "rerank_rank"
            ] = rank

            selected.append(
                document
            )

        return selected


# ============================================================
# RETRIEVAL + RERANKING
# ============================================================

def retrieve_and_rerank(
    inputs,
    retriever,
    reranker
):

    question = inputs[
        "question"
    ]

    # --------------------------------------------------------
    # FAISS TOP-5
    # --------------------------------------------------------

    candidate_documents = (
        retriever.invoke(
            question
        )
    )

    # --------------------------------------------------------
    # CROSSENCODER TOP-3
    # --------------------------------------------------------

    documents = reranker.rerank(
        question,
        candidate_documents
    )

    # --------------------------------------------------------
    # FORMAT CONTEXT
    # --------------------------------------------------------

    context = format_documents(
        documents
    )

    return {

        "question": question,

        "candidate_docs": (
            candidate_documents
        ),

        "documents": documents,

        "context": context
    }


# ============================================================
# BUILD RAG CHAIN
# ============================================================

def build_rag_chain(
    embedding_model,
    faiss_retriever,
    dataframe,
    llm_generator
):

    # --------------------------------------------------------
    # LANGCHAIN RETRIEVER
    # --------------------------------------------------------

    retriever = AlbertFaissRetriever(

        embedding_model=embedding_model,

        faiss_retriever=faiss_retriever,

        dataframe=dataframe,

        k=FAISS_TOP_K
    )

    # --------------------------------------------------------
    # RERANKER
    # --------------------------------------------------------

    reranker = DocumentReranker(
        top_k=RERANK_TOP_K
    )

    # --------------------------------------------------------
    # RETRIEVAL PIPELINE
    # --------------------------------------------------------

    retrieval_chain = RunnableLambda(
        lambda inputs:
            retrieve_and_rerank(
                inputs,
                retriever,
                reranker
            )
    )

    # --------------------------------------------------------
    # GENERATION PIPELINE
    # --------------------------------------------------------

    answer_chain = (

        {
            "context": itemgetter(
                "context"
            ),

            "question": itemgetter(
                "question"
            )
        }

        | MEDICAL_PROMPT

        | RunnableLambda(
    lambda prompt: llm_generator.generate_from_prompt(
        prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
    )
)
    )

    # --------------------------------------------------------
    # COMPLETE CHAIN
    # --------------------------------------------------------

    rag_chain = (

        retrieval_chain

        | RunnablePassthrough.assign(
            answer=answer_chain
        )
    )

    return rag_chain