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
# from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

def embed_file(file):
    st.write(file)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
        
    #여기서부턴 지난 수업내용 copy-paste
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, callbacks=[LangChainTracer(client=Client())])

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

    #각 문서에서 부분 답변을 생성하는 체인
    map_doc_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following portion of a long document to see if any of the text is relevant to answer the question. Return any relevant text verbatim.\n------\n{portion}"),
        ("human","{question}")
    ])
    map_doc_chain = map_doc_prompt | llm

    #모든 doc에 대해 map_doc_chain을 invoke -> 결과를 하나의 문자열로 만들어서 반환
    def map_docs(inputs):
        documents = inputs["documents"]
        question = inputs["question"]
        return "\n\n".join(map_doc_chain.invoke({"portion":doc.page_content, "question":question}).content for doc in documents)

    #retriever가 가져온 문서들을 map_docs에 넣어 -> 각각의 문서에 대한 답변들을 합해서 하나의 text를 만듦
    retriever = vectorstore.as_retriever()
    return retriever


st.title("Documnet GPT")

st.markdown("""
Welcome!
            
Use this chatbot to ask questions to an AI about your files!            
""")

file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"])

if file:
    retriever = embed_file(file) #사용자가 뭔가 입력할 때마다 전체 함수가 실행됨 -> embeddings가 cache가 돼있어도 시간이 걸림.
    s = retriever.invoke("winston")
    st.write(s)