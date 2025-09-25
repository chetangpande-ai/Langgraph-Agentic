# 📄 LangChain RAG Pipeline with FAISS, HuggingFace & Gemini

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using:

- **LangChain** for orchestration  
- **PyPDFLoader** to read PDF documents  
- **RecursiveCharacterTextSplitter** to chunk text  
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) for vectorization  
- **FAISS** as the vector database  
- **Google Gemini (via LangChain)** as the LLM  
- **LangChain Hub prompt** for RAG template  

---

## ⚙️ Features
- Load and process PDFs into vector embeddings  
- Store embeddings in FAISS vector store  
- Retrieve relevant chunks with **MMR (Maximal Marginal Relevance)**  
- Use Gemini to answer questions based on retrieved context  
- Clean, optimized pipeline  

---

🧠 How It Works

Load PDF → PyPDFLoader reads the file.

Chunk Text → RecursiveCharacterTextSplitter breaks content into manageable pieces.

Embeddings → HuggingFaceEmbeddings converts chunks into 384-dim vectors.

Store in FAISS → vector DB optimized for similarity search.

Retriever → MMR ensures diverse + relevant results.

Prompt + LLM → Inject retrieved context into a RAG prompt, query Gemini, and parse output.

📌 Example Query
query = "When is Safety in Pretraining llama model"
result = rag_chain.invoke(query)
print(result)

🚀 Future Enhancements

Persist FAISS index to disk (avoid re-embedding every run)

Add support for multiple PDFs

Integrate UI (e.g., Streamlit or Gradio)

Try other LLMs (OpenAI, Claude, Mistral, etc.)
