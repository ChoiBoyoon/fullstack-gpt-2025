import streamlit as st
import time

st.title("Documnet GPT")

# with st.chat_message("human"):
#     st.write("Hello!")

# with st.chat_message("ai"): #chat_message의 파라미터 name : "user", "assistant", "ai", "human", or str
#     st.write("how are you?")

# with st.status("Embedding file....", expanded=True) as status:
#     time.sleep(2)
#     st.write("Getting the file")
#     time.sleep(2)
#     st.write("Embedding the file")
#     time.sleep(2)
#     st.write("Caching the file")
#     status.update(label="Error", state="error")

# st.chat_input("Send a message to the AI")

if "messages" not in st.session_state:
    st.session_state["messages"] = [] #session state를 이용하면 whole code가 다시 실행되더라도 이 안의 데이터는 살아있음(cached). 단 새로고침을 하면 사라짐.

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message":message, "role":role})

for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)

message = st.chat_input("Send a message to the AI")

if message:
    send_message(message, "human") #우리가 메시지를 보낼 때마다 대화의 모든 messages를 다시 그림 (캐시로부터)
    time.sleep(1)
    send_message(f"You said {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)