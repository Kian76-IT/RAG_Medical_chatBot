import os
from setting import DATA_PATH
import numpy as np

from primary.load import load_data
from models.base import BaseEmbeddingModel

from rag.retriever import Retriever
from rag.pipeline import run_rag
from models.albert import AlbertModel

# Pastikan import generator sudah benar
from llm.generator import LLMGenerator

def main():
    # =====================================================
    # LOAD DATA
    # =====================================================
    print("=" * 50)
    print("Loading dataset...")
    print("=" * 50)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: File dataset tidak ditemukan di {DATA_PATH}!")
        return
        
    df, texts = load_data(DATA_PATH)
    print(f"Dataset loaded! Total data: {len(df)} baris.")

    # =====================================================
    # EMBEDDING MODEL
    # =====================================================
    print("\nLoading embedding model...")
    embedding_model = AlbertModel()

    print("Creating embeddings...")
    embeddings = np.load(
    "embedding/embeddings.npy"
    )
    print("Embeddings created successfully!")

    # =====================================================
    # RETRIEVER
    # =====================================================
    retriever = Retriever(embeddings)
    print("Retriever ready!")

    # =====================================================
    # LOAD LLM (LoRA + BASE MODEL)
    # =====================================================
    print("\nLoading LLM (Base Model + LoRA Adapter)...")
    
    # Memastikan folder LoRA hasil training Anda ada sebelum di-load
    if not os.path.exists("medical_lora_adapter"):
        print("Error: Folder 'medical_lora_adapter' tidak ditemukan!")
        print("Pastikan Anda sudah menjalankan training atau memindahkan foldernya ke sini.")
        return

    # Memanggil kelas LLMGenerator yang sudah Anda perbarui kemarin
    llm = LLMGenerator("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("LLM loaded and ready to use!")

    # =====================================================
    # CHAT LOOP
    # =====================================================
    print("\n" + "=" * 50)
    print("MEDICAL AI CHATBOT READY (Type 'exit' to quit)")
    print("=" * 50)

    while True:
        try:
            query = input("\nYou: ").strip()

            if not query:
                continue

            if query.lower() == "exit":
                print("Chat ended. Goodbye!")
                break

            # 1. PROSES RAG: RETRIEVAL
            results = run_rag(
                query,
                embedding_model,
                retriever,
                df,
                k=3 # Mengambil 3 dokumen teratas
            )

            # 2. PROSES RAG: AUGMENTATION (Menggabungkan konteks)
            context = "\n".join(results["contexts"])

            # 3. PROSES RAG & LORA: GENERATION
            # Teks query dan context dikirim ke model bertenaga LoRA Anda
            response = llm.generate(query, context)

            # OUTPUT
            print("\nBot:")
            print(response)

        except KeyboardInterrupt:
            print("\nChat ended abruptly.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()