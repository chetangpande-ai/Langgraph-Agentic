import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
#output=model.invoke("hi")
#print(output.content)

# Streamlit UI
st.set_page_config(page_title="Gemini Chat", page_icon="🤖")
st.title("🤖 Chat with Gemini (LangChain + Streamlit)")

# Maintain chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display past messages
for msg in st.session_state["messages"]:
    role, text = msg
    with st.chat_message(role):
        st.markdown(text)

# Input box for user
if prompt := st.chat_input("Type your message..."):
    # Save user message
    st.session_state["messages"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Gemini
    response = model.invoke(prompt)

    # Save model response
    st.session_state["messages"].append(("assistant", response.content))
    with st.chat_message("assistant"):
        st.markdown(response.content)