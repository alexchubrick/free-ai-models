import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

def load_api_key():
    load_dotenv()
    return os.getenv('GROK_API_KEY')

def configure_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def generate_story(model):
    return model.generate_content("Write an upbeating story about a person who stopped being a recluse in a modern age era.")

def main():
    api_key = load_api_key()
    model = configure_model(api_key)
    response = generate_story(model)
    st.write(response.text)

if __name__ == "__main__":
    main()

st.input()