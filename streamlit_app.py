import streamlit as st
from langchain_community.llms.deepseek import DeepSeek
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Function to generate response using DeepSeek
def generate_response(txt, deepseek_api_key):
    # Instantiate DeepSeek LLM with API details
    llm = DeepSeek(
        model="deepseek-chat",  # Specify DeepSeek model
        deepseek_api_key=deepseek_api_key,  # User-provided API key
        base_url="https://api.deepseek.com/v1"  # DeepSeek API endpoint
    )

    # Split text into chunks
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)

    # Create multiple document objects
    docs = [Document(page_content=t) for t in texts]

    # Perform text summarization using LangChain's summarize chain
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

# Page title
st.set_page_config(page_title='🦜🔗 DeepSeek Text Summarization App')
st.title('🦜🔗 DeepSeek Text Summarization App')

# Text input field
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user API key and process text
result = []
with st.form('summarize_form', clear_on_submit=True):
    deepseek_api_key = st.text_input('DeepSeek API Key', type='password', disabled=not txt_input)
    submitted = st.form_submit_button('Submit')

    if submitted and deepseek_api_key:
        with st.spinner('Generating summary...'):
            response = generate_response(txt_input, deepseek_api_key)
        result.append(response)

if len(result):
    st.info(response)

# Instructions to get a DeepSeek API key
st.subheader("Get a DeepSeek API Key")
st.write("You can get your own DeepSeek API key by following these steps:")
st.write("""
1. Go to [DeepSeek AI](https://deepseek.com/).
2. Sign up or log in to your account.
3. Navigate to the API section and generate a new API key.
4. Copy the key and paste it above to use this app.
""")
