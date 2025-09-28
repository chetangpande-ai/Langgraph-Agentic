"""
agentic_flow_example1.py

A runnable Python script converted from `agentic_flow_example1.ipynb`.

Features implemented:
- Load environment variables (from .env)
- Initialize Gemini (Google Generative AI) model via langchain_google_genai
- Initialize HuggingFace embeddings (BAAI/bge-large-en-v1.5)
- Load text documents from a directory and split them into chunks
- Create a Chroma vector store from documents and embeddings
- Provide a simple interactive console query example

Usage:
1. Create a `.env` file with your Google API key:
   GOOGLE_API_KEY=your_key_here
2. Put text files under a `data/` folder (or change DATA_DIR constant)
3. Run:
   python agentic_flow_example1.py

Note: This script assumes the relevant libraries are installed:
  pip install langchain langchain-google-genai langchain-huggingface langchain-community chromadb python-dotenv
(Adjust package names if necessary for your environment)
"""

import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Constants
DATA_DIR = "data"  # change to your data folder (relative to where you run the script)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
GEMINI_MODEL = "gemini-2.5-flash"  # change if you want another Gemini model

def init_gemini_model():
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as e:
        raise ImportError("Please install langchain_google_genai (or ensure langchain and Google genai support installed).") from e

    # Ensure API key is present
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("WARNING: GOOGLE_API_KEY not set. Set it in your .env file or environment before calling Gemini.")
    model = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    return model

def init_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception as e:
        raise ImportError("Please install langchain_huggingface (or ensure appropriate package installed).") from e
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeddings

def load_and_split_documents(data_dir=DATA_DIR):
    """
    Loads text files from data_dir and splits them into chunks.
    Returns a list of LangChain Document objects.
    """
    try:
        from langchain_community.document_loaders import TextLoader, DirectoryLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception as e:
        raise ImportError("Please install langchain-community and langchain (for document loaders & text splitter).") from e

    if not os.path.isdir(data_dir):
        print(f"Data directory '{data_dir}' not found. Create it and add some .txt files (or change DATA_DIR).")
        return []

    loader = DirectoryLoader(data_dir, glob="./*.txt", loader_cls=TextLoader)
    docs = loader.load()
    if not docs:
        print("No documents loaded from the data directory.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    new_docs = text_splitter.split_documents(documents=docs)
    return new_docs

def build_vector_store(docs, embeddings):
    """
    Build a Chroma vector store from documents and embeddings. Returns the Chroma DB instance and retriever.
    """
    if not docs:
        raise ValueError("No documents provided to build vector store.")

    # Prefer community package import; fallback to langchain.vectorstores if needed
    try:
        from langchain_community.vectorstores import Chroma
    except Exception:
        try:
            from langchain.vectorstores import Chroma
        except Exception as e:
            raise ImportError("Please install an appropriate Chroma vectorstore integration (langchain-community or langchain).") from e

    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return db, retriever

def example_query(retriever=None, model=None):
    """
    Demonstrates retrieval and optional generation with Gemini.
    """
    if retriever is None:
        print("No retriever provided. Skipping retrieval example.")
        return

    query = input("Enter a query to run against the vector store (or press Enter to skip): ").strip()
    if not query:
        print("No query entered; skipping.")
        return

    # Retrieve top docs
    try:
        results = retriever.get_relevant_documents(query)
    except Exception:
        # Some community retriever wrappers use `invoke`
        try:
            results = retriever.invoke(query)
        except Exception as e:
            print("Failed to retrieve documents:", e)
            return

    print(f"Retrieved {len(results)} documents (showing preview):")
    for i, r in enumerate(results[:3], start=1):
        preview = getattr(r, "page_content", str(r))[:400].replace("\n", " ")
        print(f"\n--- Doc {i} preview ---\n{preview}\n")

    if model:
        # Build a short prompt with context
        context = "\n\n".join([getattr(d, "page_content", "") for d in results[:3]])
        prompt = f"You are a helpful assistant. Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        try:
            response = model.invoke(prompt)
            print("\n--- Gemini Response ---\n")
            print(response.content)
        except Exception as e:
            print("Failed to call Gemini model:", e)
    else:
        print("No Gemini model provided, retrieval-only demo finished.")

def main():
    print("=== Agentic Flow Example (script) ===")
    # Initialize components
    embeddings = None
    retriever = None
    model = None

    try:
        embeddings = init_embeddings()
        print("Embeddings initialized.")
    except Exception as e:
        print("Embeddings init error (continuing):", e)

    docs = []
    try:
        docs = load_and_split_documents()
        print(f"Loaded and split {len(docs)} document chunks.")
    except Exception as e:
        print("Document loading error (continuing):", e)

    if embeddings and docs:
        try:
            db, retriever = build_vector_store(docs, embeddings)
            print("Chroma vector store built.")
        except Exception as e:
            print("Vector store build error (continuing):", e)

    # Initialize Gemini model (optional; requires GOOGLE_API_KEY)
    try:
        model = init_gemini_model()
        print("Gemini model initialized.")
    except Exception as e:
        print("Gemini init error (continuing):", e)

    # Interactive query/demo
    example_query(retriever=retriever, model=model)
    print("Script finished.")

if __name__ == "__main__":
    main()
