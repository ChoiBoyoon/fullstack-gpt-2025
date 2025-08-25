# test URL
# https://openai.com/index/introducing-gpts/
# https://openai.com/sitemap.xml

# from langchain.document_loaders import AsyncChromiumLoader
# from langchain.document_transformers import Html2TextTransformer #외부 package지만 LangChain에서 내부적으로 사용
from langchain.document_loaders import SitemapLoader
import streamlit as st

st.set_page_config(page_title="SiteGPT", page_icon="🌐")
st.title("SiteGPT")
st.markdown("""
        Welcome to SiteGPT!
            
        test URL : https://openai.com/index/introducing-gpts/

        test URL2: https://openai.com/sitemap.xml
""")

# html2text_transformer = Html2TextTransformer()

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    loader = SitemapLoader(url)
    loader.requests_per_second = 1 #openai의 경우 951개의 페이지. 차단 방지를 위해 1초에 1개의 페이지를 스크랩함. 변경 가능.
    docs = loader.load() #load_and_split()도 있음.
    return docs
with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.") #sitemap의 URL들을 구글 등 크롤러의 스크랩을 허용해놓은 것들.
    else:
        docs=load_website(url)
        st.write(docs)

    # #async chromium loader. 완전한 browser를 실행 중 - 느려질 수 있음
    # loader = AsyncChromiumLoader([url]) #takes a list of urls but we receive only one url
    # docs = loader.load() #list of docs
    # transformed = html2text_transformer.transform_documents(docs)
    # st.write(transformed)


