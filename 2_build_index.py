import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build():
    if not os.path.exists("data/rag_source.txt"):
        print("❌ 找不到 data/rag_source.txt")
        return

    print("📚 讀取資料並建立索引...")
    loader = TextLoader("data/rag_source.txt", encoding="utf-8")
    docs = CharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(loader.load())
    
    print("🧠 下載向量模型中 (第一次會比較久)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    print("✅ 索引建立完成！資料夾: faiss_index")

if __name__ == "__main__":
    build()