# test URL
# https://openai.com/index/introducing-gpts/
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer #외부 package지만 LangChain에서 내부적으로 사용
import streamlit as st

st.set_page_config(page_title="SiteGPT", page_icon="🌐")

st.title("SiteGPT")
html2text_transformer = Html2TextTransformer()

st.markdown("""
        Welcome to SiteGPT!
            
        test URL : https://openai.com/index/introducing-gpts/
""")

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

if url:
    #async chromium loader
    loader = AsyncChromiumLoader([url]) #takes a list of urls but we receive only one url
    docs = loader.load() #list of docs
    transformed = html2text_transformer.transform_documents(docs)
    st.write(transformed)


