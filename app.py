import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import Document
import validators

# Set up the Streamlit app configuration
st.set_page_config(page_title="Website Summarizer", page_icon="🌐")
st.title("🌐 Website Summarizer")

# Sidebar for HF API Key
with st.sidebar:
    hf_api_key=st.text_input("Huggingface API Token",value="",type="password")

# Main section for website summarization
st.subheader("Enter Website URL")
website_url = st.text_input("Website URL", placeholder="e.g., https://www.example.com")

# Gemma Model Using Groq API
repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temperature=0.7,token=hf_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def fetch_website_content(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        content = "\n".join([para.get_text() for para in paragraphs])
        return content
    except Exception as e:
        st.error(f"An error occurred while fetching the website content: {e}")
        return None

if st.button("Summarize the Website"):
    # Validate the inputs
    if not hf_api_key.strip() or not website_url.strip():
        st.error("Please provide the API key and website URL.")
    elif not validators.url(website_url):
        st.error("Please enter a valid website URL.")
    else:
        try:
            with st.spinner("Processing..."):
                # Fetch and parse the website content
                content = fetch_website_content(website_url)
                
                if not content:
                    st.error("No content could be loaded from the provided URL.")
                else:
                    # Wrap the content in a Document object
                    doc = Document(page_content=content)
                    
                    # Summarize the content
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run([doc])
                    
                    # Display the summary
                    st.success("Summary:")
                    st.write(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")