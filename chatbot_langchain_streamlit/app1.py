
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Page configuration
st.set_page_config(
    page_title="Gemini Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #e3f2fd, #fce4ec);
}

/* Header logo and title */
.header {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 20px;
}
.header img {
    width: 50px;
    height: 50px;
}

/* Chat bubbles */
.chat-bubble-user {
    background-color: #2196F3;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
    align-self: flex-end;
}
.chat-bubble-assistant {
    background-color: #f1f1f1;
    color: #000;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
    align-self: flex-start;
}

/* Role label */
.role-label {
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 2px;
}

/* Chat container */
.chat-container {
    display: flex;
    flex-direction: column;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 12px;
    color: gray;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# --- Header with logo ---
st.markdown("""
<div class="header">
    <img src="https://img.icons8.com/ios-filled/50/000000/robot-2.png"/>
    <h2>🤖 Gemini Chatbot</h2>
</div>
""", unsafe_allow_html=True)

st.write("How can I assist today?")

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat
chat_container = st.container()
with chat_container:
    for role, text in st.session_state["messages"]:
        label = "You" if role == "user" else "Gemini"
        bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
        st.markdown(f"""
        <div class='chat-container'>
            <div class='role-label'>{label}</div>
            <div class='{bubble_class}'>{text}</div>
        </div>
        """, unsafe_allow_html=True)

# User input box
if prompt := st.chat_input("Type your message here..."):
    # Save user message
    st.session_state["messages"].append(("user", prompt))
    st.markdown(f"""
    <div class='chat-container'>
        <div class='role-label'>You</div>
        <div class='chat-bubble-user'>{prompt}</div>
    </div>
    """, unsafe_allow_html=True)

    # Get Gemini response
    response = model.invoke(prompt)
    st.session_state["messages"].append(("assistant", response.content))
    st.markdown(f"""
    <div class='chat-container'>
        <div class='role-label'>Gemini</div>
        <div class='chat-bubble-assistant'>{response.content}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class='footer'>
    © 2025 Gemini Chatbot. All rights reserved. | Dummy footer text
</div>
""", unsafe_allow_html=True)
