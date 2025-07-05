from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import streamlit as st

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Streamlit interface
st.header('Research Assistant (FLAN-T5)')

user_input = st.text_input('Enter your prompt:')

if st.button('Search'):
    response = llm.invoke(user_input)
    st.write('Response:')
    st.write(response)
