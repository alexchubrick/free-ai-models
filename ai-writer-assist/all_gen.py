import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from mistralai import Mistral


def load_api_key(env_var):
    """
    Load the API key from environment variables.
    """
    load_dotenv()
    api_key = os.getenv(env_var)
    if not api_key:
        st.error(f"API key for {env_var} is not set.")
        st.stop()
    return api_key


def configure_model(api_key, model_name):
    """
    Configure the appropriate model based on the selection.
    """
    if model_name.startswith('Gemini 1.5'):
        genai.configure(api_key=api_key)
        model_map = {
            'Gemini 1.5 Flash': "gemini-1.5-flash",
            'Gemini 1.5 Flash-8B': "gemini-1.5-flash-8b",
            'Gemini 1.5 Pro': "gemini-1.5-pro"
        }
        return genai.GenerativeModel(model_map[model_name])
    elif model_name.startswith('GPT-4o'):
        client = OpenAI(api_key=api_key)
        return client
    elif model_name.startswith('Grok'):
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        return client
    else:
        st.error(f"Unsupported model: {model_name}")
        st.stop()


def generate_answer(model, prompt, show_in_chunks, model_name):
    """
    Generate the response from the selected model.
    """
    if model_name.startswith('Gemini 1.5'):
        stream = show_in_chunks
        response = model.generate_content(prompt, stream=stream)
        return response
    elif model_name.startswith('GPT-4o'):
        completion = model.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return completion
    elif model_name.startswith('Grok'):
        completion = model.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": "You are Grok, a helpful chatbot producing thorough and actionable answers to all questions."},
                {"role": "user", "content": prompt},
            ],
        )
        return completion
    else:
        st.error("Failed to generate a response.")
        return None


def main(model_name, prompt, show_in_chunks):
    """
    Main logic to handle model selection, API configuration, and response generation.
    """
    api_key_env_var = {
        'Gemini': 'GEMINI_API_KEY',
        'GPT': 'GPT_API_KEY',
        'Grok': 'GROK_API_KEY',
        'Mistral': 'MISTRAL_API_KEY',
        'Cohere': 'COHERE_API_KEY',
        'Gemma': 'GEMMA_API_KEY',
        'Llama': 'LLAMA_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY',
        'Goose': 'GOOSE_API_KEY'
    }.get(next((key for key in ['Gemini', 'GPT', 'Grok', 'Mistral', 'Cohere', 'Gemma', 'Llama', 'Anthropic', 'Goose'] if model_name.startswith(key)), None), None)
    api_key = load_api_key(api_key_env_var)
    model = configure_model(api_key, model_name)

    response = generate_answer(model, prompt, show_in_chunks, model_name)
    if response:
        if show_in_chunks and model_name.startswith('Gemini 1.5'):
            for chunk in response:
                st.write(chunk.text)
        else:
            st.write(response.text if model_name.startswith(
                'Gemini 1.5') else response.choices[0].message.content)


# Streamlit interface
with st.form("generator"):
    chosen_model = st.selectbox(
        'Choose a model:',
        ['Gemini 1.5 Flash',
         'Gemini 1.5 Flash-8B',
         'Gemini 1.5 Pro',
         'Grok',
         'Mistral',
         'Cohere',
         'GPT-4o',
         'GPT-4o mini']
    )
    user_prompt = st.text_input('Type your prompt:')
    show_in_chunks = st.checkbox('Show the results in chunks')
    st.markdown(
        '- :gray[You can wait for the full response or start getting it immediately in smaller chunks.]')
    submit = st.form_submit_button('Generate')

if submit:
    if not user_prompt.strip():
        st.error("Prompt cannot be empty. Please enter a valid prompt.")
    else:
        st.markdown(f'**Chosen model**: {chosen_model}')
        st.markdown(f'**Your prompt**: {user_prompt}')
        st.write("-" * 80)
        main(chosen_model, user_prompt, show_in_chunks)
