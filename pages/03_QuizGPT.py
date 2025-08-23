import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import json

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```","").replace("json","") #prompt engineering의 한 부분. output의 특정 형식을 요청할 때 넣으면 쓸데없는 말 안하고 아주 잘 작동함. ("그럼요! 아래는 요청한 형식입니다" 같은 말들..)
        return json.loads(text) #str을 Python object로 바꿔줌
output_parser=JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

llm = ChatOpenAI(
    model="chatgpt-4o-latest", #"gpt-5-2025-08-07"은 기업 계정에만 streaming을 제공
    streaming=True,
    callbacks = [StreamingStdOutCallbackHandler()]
    # temperature=0.1, #이 모델에선 더 이상 지원되지 않음. 오직 1만 가능
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) #retriever가 가져온 docs를 하나의 str으로 만듦

questions_prompt = ChatPromptTemplate.from_messages([
    ("system",
        """
    You are a helpful assistant that is role playing as a teacher.
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text. 
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    Use (o) to signal the correct answer.
    
    Question examples:

        
    Your turn!

    Context: {context}
    """)
])

questions_chain = {"context":format_docs}|questions_prompt|llm

formatting_prompt = ChatPromptTemplate.from_messages([
    ('system',"""
    You are a powerful formatting algorithm.
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
    
    Example Input:
    
    Question: What is the color of the ocean?
    Answer: Red| Yellow| Green| Blue(o)

    Question: What is the capital of Georgia?
    Answer: Baku| Tbilisi(o)| Manila| Beirut

    Question: When was Avatar released?
    Answer: 2007| 2001| 2009(o)| 1998

    Question: Who was Julius Caesar?
    Answer: A Roman Emperor(o)| Painter| Actor| Model
    
    ```json
    {{"questions": [
    {{
    "question": "What is the color of the ocean?",
    "answers": [
    {{
    "answer": "Red",
    "correct": false
    }},
    {{
    "answer": "Yellow",
    "correct": false
    }},
    {{
    "answer": "Green",
    "correct": false
    }},
    {{
    "answer": "Blue",
    "correct": true
    }},
    ]
    }},
    {{
    "question": "What is the capital or Georgia?",
    "answers": [
    {{
    "answer": "Baku",
    "correct": false
    }},
    {{
    "answer": "Tbilisi",
    "correct": true
    }},
    {{
    "answer": "Manila",
    "correct": false
    }},
    {{
    "answer": "Beirut",
    "correct": false
    }},
    ]
    }},
    {{
    "question": "When was Avatar released?",
    "answers": [
    {{
    "answer": "2007",
    "correct": false
    }},
    {{
    "answer": "2001",
    "correct": false
    }},
    {{
    "answer": "2009",
    "correct": true
    }},
    {{
    "answer": "1998",
    "correct": false
    }},
    ]
    }},
    {{
    "question": "Who was Julius Caesar?",
    "answers": [
    {{
    "answer": "A Roman Emperor",
    "correct": true
    }},
    {{
    "answer": "Painter",
    "correct": false
    }},
    {{
    "answer": "Actor",
    "correct": false
    }},
    {{
    "answer": "Model",
    "correct": false
    }},
    ]
    }}
    ]
    }}
    ```
    Your turn!
    Questions: {context}
    """)
])

formatting_chain = formatting_prompt|llm

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

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context":questions_chain}|formatting_chain|output_parser
    return chain.invoke(_docs)
    
@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=1, lang='en') # 상위 k개 문서만 가져옴. 안그러면 해당 키워드가 포함된 많은 페이지를 가져올 수 있음.
    docs = retriever.get_relevant_documents(term)
    return docs
    

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
            docs = wiki_search(topic)
            

if not docs:
    st.markdown(
        """
        Welcome to QuizGPT
        
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
        
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name) #if topic exists, it's wiki. if topic doesn't exist, it's file
    with st.form("questions_form"): #모든 form에는 submit 버튼이 필요
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio("Select one answer", 
                     [answer["answer"] for answer in question["answers"]], 
                     index=None) #no default selection
            if {"answer":value, "correct":True} in question["answers"]:
                st.success("Correct")
            elif value is not None:
                st.error("Wrong")
        button = st.form_submit_button()

