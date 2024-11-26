import streamlit as st
from yandex_cloud_ml_sdk import YCloudML

sdk = YCloudML(folder_id="b1g7be5bgf5k516ljupc",
               auth='AQVNz0fPwnHlsknkIeZxNPOoWd60KKw7OjSsTUIa')

model = sdk.models.completions('yandexgpt')
model = model.configure(temperature=0.5)
result = model.run("Что такое небо?")

st.write(result[0].text)
