import streamlit as st
from transformers import pipeline

# Initialize the pipeline
pipe = pipeline("text-generation", model='gpt2-medium')


def generate_summary(sample_text, pipe, max_length=512):
    query = sample_text + "\nTL;DR:\n"
    pipe_out = pipe(query, max_length=max_length, clean_up_tokenization_spaces=True, truncation=True)
    summary = pipe_out[0]['generated_text'][len(query):]
    return summary

# Streamlit app
st.title("Text Summary Generator with 'gpt2-medium' Hugging Face")

# Text input
input_text = st.text_area("Enter text to summarize:", height=300)

# Button to generate summary
if st.button("Generate Summary"):
    with st.spinner("Generating summary..."):
        summary = generate_summary(input_text, pipe)
        st.success("Summary generated!")
        st.subheader("Summary")
        st.write(summary)
        st.subheader("Original Text Length")
        st.write(len(input_text))
        st.subheader("Summary Text Length")
        st.write(len(summary))
