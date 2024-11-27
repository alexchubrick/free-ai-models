import os

from dotenv import load_dotenv
import anthropic
import cohere
from huggingface_hub import InferenceClient
from mistralai import Mistral
from openai import OpenAI
import streamlit as st
import google.generativeai as genai
from yandex_cloud_ml_sdk import YCloudML


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
        return OpenAI(api_key=api_key)
    elif model_name.startswith('Grok'):
        return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    elif model_name.startswith('Mistral'):
        return Mistral(api_key=api_key)
    elif model_name.startswith('Cohere'):
        return cohere.ClientV2(api_key=api_key)
    elif model_name.startswith('Gemma'):
        return InferenceClient(api_key=api_key)
    elif model_name.startswith('Llama'):
        return OpenAI(api_key=api_key, base_url="https://api.llama-api.com")
    elif model_name.startswith('Anthropic'):
        return anthropic.Anthropic()
    elif model_name.startswith('YandexGPT'):
        sdk = YCloudML(folder_id="b1g7be5bgf5k516ljupc", auth=api_key)
        return sdk.models.completions('yandexgpt').configure(temperature=0.5)
    else:
        st.error(f"Unsupported model: {model_name}")
        st.stop()


def generate_answer(model, prompt, model_name):
    """
    Generate the response from the selected model.
    """
    if model_name.startswith('Gemini 1.5'):
        response = model.generate_content(prompt)
        return response
    elif model_name.startswith('GPT-4o') or model_name.startswith('Grok'):
        completion = model.chat.completions.create(
            model="gpt-4o-mini" if model_name.startswith(
                'GPT-4o') else "grok-beta",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion
    elif model_name.startswith('Mistral'):
        return model.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}]
        )
    elif model_name.startswith('Cohere'):
        return model.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": prompt}]
        )
    elif model_name.startswith('Gemma'):
        return model.chat.completions.create(
            model="google/gemma-2-2b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
    elif model_name.startswith('Llama'):
        return model.chat.completions.create(
            model="llama3.1-70b",
            messages=[
                {"role": "system",
                    "content": "Assistant is a large language model trained by OpenAI."},
                {"role": "user", "content": prompt}
            ]
        )
    elif model_name.startswith('Anthropic'):
        return model.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": {
                "type": "text", "text": prompt}}]
        )
    elif model_name.startswith('YandexGPT'):
        return model.run(prompt)
    else:
        st.error("Failed to generate a response.")
        return None


def main(model_name, prompt):
    """
    Main logic to handle model selection, API configuration, and response generation.
    """
    api_key_env_var = {
        'Gemini': 'GEMINI_API_KEY',
        'GPT': 'GPT_API_KEY',
        'Grok': 'GROK_API_KEY',
        'Mistral': 'MISTRAL_API_KEY',
        'Cohere': 'COHERE_API_KEY',
        'Gemma': 'HUGGING_FACE_API_KEY',
        'Llama': 'LLAMA_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY',
        'YandexGPT': 'YANDEX_API_KEY'
    }.get(next((key for key in ['Gemini', 'GPT', 'Grok', 'Mistral', 'Cohere', 'Gemma', 'Llama', 'Anthropic', 'YandexGPT'] if model_name.startswith(key)), None), None)
    api_key = load_api_key(api_key_env_var)
    model = configure_model(api_key, model_name)

    response = generate_answer(model, prompt, model_name)
    if response:
        if model_name.startswith('Cohere'):
            st.write(response.message.content[0].text)
        elif model_name.startswith('YandexGPT'):
            st.write(response[0].text)
        else:
            st.write(response.text if hasattr(response, 'text')
                     else response.choices[0].message.content)


# Streamlit interface
st.header("Your Favorite _AI Assistants_ :blue[For Free] :sunglasses:")

with st.form("generator"):
    chosen_model = st.selectbox(
        'Choose a model:',
        ['Gemini 1.5 Flash', 'Gemini 1.5 Flash-8B', 'Gemini 1.5 Pro', 'Gemma',
         'Grok', 'Mistral', 'Cohere', 'YandexGPT', 'GPT-4o (inactive)', 'Llama (inactive)', 'Anthropic (inactive)']
    )
    user_prompt = st.text_input('Type your prompt:')
    submit = st.form_submit_button('Generate')

if submit:
    if not user_prompt.strip():
        st.error("Prompt cannot be empty. Please enter a valid prompt.")
    else:
        st.markdown(f'**Chosen model**: {chosen_model}')
        st.markdown(f'**Your prompt**: {user_prompt}')
        st.write("-" * 80)
        main(chosen_model, user_prompt)
