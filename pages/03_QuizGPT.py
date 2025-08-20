import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

llm = ChatOpenAI(
    model="gpt-5-2025-08-07",
    streaming=True,
    callbacks = [StreamingStdOutCallbackHandler()]
    # temperature=0.1, #이 모델에선 더 이상 지원되지 않음. 오직 1만 가능
)

@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    # st.write(file)
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
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
    return docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) #retriever가 가져온 docs를 하나의 str으로 만듦

with st.sidebar:
    docs=None
    choice = st.selectbox("Choose what you want to use", (
        "File","Wikipedia Article"
    ))
    if choice=="File": #여기선 embed, vectorstore등을 이용하지 않을거. 저렴하게 구동하려면 작은 파일을 사용하시오
        file=st.file_uploader("Upload a .docx, .txt or .pdf file", type=["pdf","txt","docx"])
        if file:
            docs=split_file(file)
    else:
        topic = st.text_input("Search Wikipedia")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5) # 상위 k개 문서만 가져옴. 안그러면 해당 키워드가 포함된 많은 페이지를 가져올 수 있음.
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)

if not docs:
    st.markdown(
        """
        Welcome to QuizGPT
        
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
        
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
else:
    st.write(docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system":"""
        You are a helpful assistant that is role playing as a teacher.
        Based ONLY on the following context make 10 questions to test the user's knowledge about the text. 
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
        Use (o) to signal the correct answer.
        
        Question examples:

         Question: What is the color of the ocean?
         Answer: Red, Yellow, Green, Blue(o)

         Question: What is the capital of Georgia?
         Answer: Baku, Tbilisi(o), Manila, Beirut
        
         Question: When was Avatar released?
         Answer: 2007, 2001, 2009(o), 1998

         Question: Who was Julius Caesar?
         Answer: A Roman Emperor(o), Painter, Actor, Model

        Your turn!
         
        Context: {context}
        """)
    ])

    chain = {"context":format_docs}|prompt|llm

    start = st.button("Generate Quiz")

    if start:
        chain.invoke(docs)

