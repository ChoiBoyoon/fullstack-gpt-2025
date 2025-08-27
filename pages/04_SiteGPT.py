# test URL
# https://openai.com/index/introducing-gpts/
# https://openai.com/sitemap.xml

# from langchain.document_loaders import AsyncChromiumLoader
# from langchain.document_transformers import Html2TextTransformer #외부 package지만 LangChain에서 내부적으로 사용
from langchain.document_loaders import SitemapLoader #internally uses BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

llm = ChatOpenAI()

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
    Then, give a score to the answer between 0 and 5. 0 being not helpful to the user and 5 being helpful to the user.

    Examples:

    Question: How far away is the moon?
    Anwer: The moon is 384,400km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know.
    Score: 0                             

    Your turn!

    Context: {context}
    Question: {question}                 
    """)

def get_answers(inputs):
    st.write("get_answers function is called")
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm
    return {"question":question, 
        "answers": [
        {
            "answer":answers_chain.invoke({"question":question, "context":doc.page_content}).content,
            "source":doc.metadata["source"],
            "data":doc.metadata["lastmod"]
        } for doc in docs]}
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke({"question":question, "context":doc.page_content})
    #     answers.append(result.content)
    # st.write(answers)

choose_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Use ONLY the following pre-existing answers to answer the user's question.
        Use the answers that have the highest score (more helpful) and favor the most recent ones.
        Return the sources of the answers as they are, do not change them.
        Answers: {answers}
    """),
    ("human","{question}")
])

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm 
    condensed = "\n\n".join(f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n" for answer in answers)
    choose_chain.invoke({"question":{question}, "answers":condensed})

st.set_page_config(page_title="SiteGPT", page_icon="🌐")
st.title("SiteGPT")
st.markdown("""
        Welcome to SiteGPT!
            
        test URL : https://openai.com/index/introducing-gpts/

        test URL2: https://openai.com/sitemap.xml
""")

# html2text_transformer = Html2TextTransformer()

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header: #soup로부터 header&footer 제거
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace('\n', ' ').replace("\xa0", ' ').replace('CloseSearch Submit Blog', ' ') #반환된 값은 page_content로서 document에 포함됨

@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    loader = SitemapLoader(
        url, 
        filter_urls=[r"^(.*\/index\/).*"], #filter_urls에는 특정 url을 넣거나 regex를 사용할 수 있음.
                                        # r"^(.*\/blog\/).*"] : 모든 blog url을 스크랩
                                        # r"^(?!.*\/blog\/).*"] : blog를 제외한 모든 url을 스크랩
        parsing_function = parse_page #sitemap의 모든 url에 대해 실행됨
    )
    loader.requests_per_second = 1 #openai의 경우 951개의 페이지. 차단 방지를 위해 1초에 1개의 페이지를 스크랩함. 변경 가능.
    docs = loader.load_and_split(text_splitter=splitter) #load() 아니면 load_and_split()
    print("Doc lengthgs: ", len(docs))
    return FAISS.from_documents(docs, OpenAIEmbeddings()).as_retriever()
    # return vector_store.as_retriever()


with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.") #sitemap의 URL들을 구글 등 크롤러의 스크랩을 허용해놓은 것들.
    else:
        retriever=load_website(url)

        query = st.text_input("Ask a question to the website.")
        if query:
            
            # 첫 번째 chain은 모든 개별 doc에 대한 답변 생성 & 채점
            # 두 번째 chain은 점수가 제일 높고 가장 최신 정보를 담고 있는 답변을 고름
            chain = {"docs":retriever, "question":RunnablePassthrough()} | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

            result = chain.invoke(query)
            st.write(result.content.replace('$','\$'))
            # docs = retriever.invoke("What is the price of the latest OpenAI model?")
        

    # #async chromium loader. 완전한 browser를 실행 중 - 느려질 수 있음
    # loader = AsyncChromiumLoader([url]) #takes a list of urls but we receive only one url
    # docs = loader.load() #list of docs
    # transformed = html2text_transformer.transform_documents(docs)
    # st.write(transformed)


