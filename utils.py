import streamlit as st
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings, OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


class ChatUtilities:
    
    @staticmethod
    def initialize_session_state():
        """세션 상태를 초기화합니다."""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            
    @staticmethod
    def save_message(message, role):
        """메시지를 세션 상태에 저장하는 함수."""
        st.session_state["messages"].append({"message": message, "role": role})

    @staticmethod
    def send_message(message, role, save=True):
        """메시지를 화면에 출력하고 세션 상태에 저장할지 여부를 선택하는 함수."""
        with st.chat_message(role):
            st.markdown(message)
        if save:
            ChatUtilities.save_message(message, role)

    @staticmethod
    def paint_history():
        """저장된 메시지 히스토리를 화면에 출력하는 함수."""
        for message in st.session_state["messages"]:
            ChatUtilities.send_message(
                message["message"],
                message["role"],
                save=False,
            )
    
    @staticmethod        
    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)
    
    @staticmethod
    @st.cache_data(show_spinner="Embedding file...")
    def embed_file(
        file, 
        embedding_class = OpenAIEmbeddings, 
        embedding_kwargs = None,
        file_folder="./.cache/files", 
        embedding_folder="./.cache/embeddings"
    ):
        
        file_path = os.path.join(file_folder, file.name)
        cache_dir = LocalFileStore(os.path.join(embedding_folder, file.name))
        
        os.makedirs(file_folder, exist_ok=True)
        os.makedirs(embedding_folder, exist_ok=True)
        
        file_content = file.read()
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        
        if embedding_kwargs is None:
            embedding_kwargs = {}
        embeddings = embedding_class(**embedding_kwargs)
        
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        
        return retriever

    
    