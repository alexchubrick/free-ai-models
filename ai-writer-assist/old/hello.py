import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from the .env file
load_dotenv()
# Access the environment variable
api_key = os.getenv('API_KEY')
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a upbeating story about a person who stopped being a recluse in a modern age era.")

st.write(response.text)