from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(
    model="gemini-2.0-flash",  
    temperature=0.7
)

st.header("RESEARCH TOOL")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

prompt_template = PromptTemplate(
    input_variables=["paper", "style", "length"],
    template="Explain the paper '{paper}' in a {style} style with a {length}."
)

if st.button('Summarize'):
    chain = prompt_template | model

    result = chain.invoke({
        "paper": paper_input,
        "style": style_input,
        "length": length_input
    })
    st.write(result)