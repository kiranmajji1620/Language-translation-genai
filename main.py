from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

def Translate(source_language, target_language, text):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id = repo_id,
        max_new_tokens=512,
        temperature = 0.1,    
    )
    prompt = PromptTemplate.from_template(f"Translate the following text from {source_language} to {target_language}: {text}")
    outputParser = StrOutputParser()
    chain = prompt|llm|outputParser
    # print(chain.input_schema.schema())
    return chain.invoke(input = {"text1" : text, "target_language" : target_language, "source_language" : source_language})

st.title("Tranlation using google t5")
source_language = st.text_input("Enter the source language")
target_language = st.text_input("Enter the target language")
text = st.text_input("Enter the text to be translated")
if source_language and target_language and text:
    st.write(Translate(source_language, target_language, text))