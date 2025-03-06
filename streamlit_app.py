import streamlit as st
from langchain.llms import DeepSeek, Anthropic  # For DeepSeek & Claude
from langchain_community.llms import Ollama  # For Local Ollama Execution
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def generate_response(txt, model, api_key=None):
    # Select the model
    if model == "DeepSeek":
        llm = DeepSeek(temperature=0, deepseek_api_key=api_key)
    elif model == "Claude":
        llm = Anthropic(model="claude-3", temperature=0, anthropic_api_key=api_key)
    elif model == "Ollama":
        llm = Ollama(model="mistral")  # Uses Ollama (Mistral model) locally
    else:
        return "Invalid model selection."
    
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    
    # Summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

# Page title
st.set_page_config(page_title='🦜🔗 Multi-AI Text Summarization App')
st.title('🦜🔗 Multi-AI Text Summarization App')

# Select Model
target_model = st.selectbox("Choose AI Model:", ["DeepSeek", "Claude", "Ollama"])

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user input
result = []
with st.form('summarize_form', clear_on_submit=True):
    if target_model in ["DeepSeek", "Claude"]:
        api_key = st.text_input(f'{target_model} API Key', type='password', disabled=not txt_input)
    else:
        api_key = None  # Ollama runs locally
    
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Summarizing...'):
            response = generate_response(txt_input, target_model, api_key)
        result.append(response)

if len(result):
    st.info(response)

# Instructions for API keys
st.subheader("API Key Instructions")
st.write("""
- **DeepSeek API Key**: Get it from [DeepSeek AI](https://platform.deepseek.com/)
- **Claude API Key**: Get it from [Anthropic](https://console.anthropic.com/)
- **Ollama**: Runs locally, no API key needed.
""")
