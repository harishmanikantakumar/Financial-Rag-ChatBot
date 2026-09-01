import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Global Cache Variables
_EMBEDDINGS = None
_VECTOR_STORE = None
_VECTOR_STORE_PATH = None


def get_embeddings():
    """Initializes and caches the Hugging Face sentence-transformers model."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        print("\n===== LOADING EMBEDDING MODEL =====")
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embedding model loaded successfully.")
    return _EMBEDDINGS


def load_documents(data_folder="data"):
    """Loads PDF documents from the specified data directory."""
    print("\n===== LOADING PDF DOCUMENTS =====")
    documents = []
    pdf_files = ["Apple.pdf", "Microsoft.pdf", "Tesla.pdf"]

    for filename in pdf_files:
        path = os.path.join(data_folder, filename)

        if not os.path.exists(path):
            lowercase_path = os.path.join(data_folder, filename.lower())
            if os.path.exists(lowercase_path):
                path = lowercase_path
            else:
                print(f"⚠️ Missing file: {path}")
                continue

        print(f"Loading: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = path
            doc.metadata["company"] = filename.replace(".pdf", "").capitalize()

        documents.extend(docs)

    print(f"Total document pages loaded: {len(documents)}")
    return documents


def create_vector_store(save_path="db/faiss_index", data_folder="data"):
    """Splits documents into text chunks, builds the FAISS index, and saves it locally."""
    print("\n" + "=" * 80)
    print("CREATING FINANCIAL FAISS VECTOR STORE")
    print("=" * 80)

    print("\nStep 1: Loading documents...")
    documents = load_documents(data_folder)
    if not documents:
        raise ValueError("No PDF documents were loaded. Check your 'data/' directory.")

    print("\nStep 2: Splitting text into chunks...")
    # FIXED: Increased chunk_size to 1500 and overlap to 300 so full financial tables stay intact
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No text chunks created after splitting.")

    print("\nStep 3: Creating FAISS vector store embeddings...")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    print("\nStep 4: Saving FAISS index locally...")
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)

    print("\n" + "=" * 80)
    print("✅ FAISS INDEX CREATED AND SAVED SUCCESSFULLY")
    print("=" * 80)
    return vector_store


def load_vector_store(faiss_path="db/faiss_index"):
    """Loads the pre-built FAISS index from disk or returns the cached instance."""
    global _VECTOR_STORE, _VECTOR_STORE_PATH

    if _VECTOR_STORE is not None and _VECTOR_STORE_PATH == faiss_path:
        print("===== USING CACHED FAISS STORE =====")
        return _VECTOR_STORE

    index_file = os.path.join(faiss_path, "index.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"FAISS index not found at '{faiss_path}'. "
            f"Please run create_vector_store() first."
        )

    print("\n===== LOADING FAISS VECTOR STORE =====")
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    _VECTOR_STORE = vector_store
    _VECTOR_STORE_PATH = faiss_path
    print("FAISS vector store loaded successfully.")
    return vector_store


if __name__ == "__main__":
    create_vector_store()