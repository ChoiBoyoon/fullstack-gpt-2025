import streamlit as st
from langsmith import Client
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(page_title="PrivateGPT", page_icon="📄")
st.title("Documnet GPT")
st.markdown("Welcome!\n\nUse this chatbot to ask questions to an AI about your files!\n\nUpload your files in the sidebar")

# @st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    # st.write(file)
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)

    # load and split
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs=loader.load_and_split(text_splitter=splitter)

    #embed, cache, and create vectorstore
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}") 
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = Chroma.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

#define callback and llm
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    def on_llm_start(self, *args, **kwargs): # can take unlimited arguments(1,2,3,..) and keyword arguments (a=5)
        self.message_box = st.empty() #빈 위젯을 제공
    def on_llm_new_token(self, token:str, *args, **kargs): #token은 LLM이 실시간으로 보내는 메시지. 토큰이 도착할 때마다 message_box에 추가 (화면에 출력됨)
        self.message += token
        self.message_box.markdown(self.message)
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.1, 
    callbacks=[LangChainTracer(client=Client()), ChatCallbackHandler()],
    streaming=True #ChatOpenAI는 지원. 다른 llm은 지원 안할 수도 있음.
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) #retriever가 가져온 docs를 하나의 str으로 만듦

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.\n\nContext: {context}"),
    ("human", "{question}")
])

#유저가 파일을 올리면 여기서부터 코드가 시작됨
with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"])

if file:
    retriever = embed_file(file) #사용자가 뭔가 입력할 때마다 전체 함수가 실행됨 -> embeddings가 cache가 돼있어도 시간이 걸림.

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history() #이 라인이 없으면 내가 새로운 질문을 할 때 이전 질문의 내용은 없어짐 (프린트되지 않음)

    message = st.chat_input("Ask questions about your file...")
    if message:
        send_message(message, "human")
        chain = {"context":retriever | RunnableLambda(format_docs), "question":RunnablePassthrough()} | prompt | llm
        # chain.invoke(message) #with없이 그냥 invoke를 하면 빈 공간에 답변이 프린트 됨 (role 없이)
        with st.chat_message("ai"):
            chain.invoke(message)

else:
    st.session_state["messages"] = [] #파일이 아직 안올라왔거나 / 파일이 없어지면(유저가 x 클릭) messages를 초기화
 