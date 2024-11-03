from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = False
if "ready_message_sent" not in st.session_state:
    st.session_state["ready_message_sent"] = False

retriever = None  # 전역적으로 retriever 변수를 초기화

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message = ""  # 메시지 초기화
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-1106",
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    try:
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return None
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message['role'],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using only the following context. If you don't know the answer just say you don't know.
            Don't make anything up.
            
            Context: {context}
            """
        ),
        ("human", "{question}")
    ]
)

st.title("Document GPT")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to an AI about your files!
    
    Upload your files on the sidebar.
    """
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    if not st.session_state["file_uploaded"]:
        retriever = embed_file(file)
        if retriever is not None:
            # 중복 메시지 방지를 위해 조건 추가
            if not st.session_state["ready_message_sent"]:
                send_message("I'm ready! Ask away!", "ai", save=True)
                st.session_state["ready_message_sent"] = True
            st.session_state["file_uploaded"] = True
        else:
            st.error("failed in making embedding, check the file")
    elif not st.session_state["ready_message_saent"]:
        paint_history()  

    message = st.chat_input("Ask anything about your file")
    if message and retriever is not None:
        send_message(message, "human")
        try:
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                chain.invoke(message)
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
else:
    # 파일이 없으면 초기화
    st.session_state["messages"] = []
    st.session_state["file_uploaded"] = False
    st.session_state["ready_message_sent"] = False