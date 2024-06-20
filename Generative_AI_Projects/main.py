import os, streamlit as st
from vish_api_keys import openaiapi # for api key, modify accordingly
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI

# LLM config
os.environ['OPENAI_API_KEY'] = openaiapi # insert your api key here
llm_openai = OpenAI(temperature=0.7, max_tokens=500) # using gpt-3.5-turbo-instruct

# Page config
st.title("URL Research Tool")
st.sidebar.title("Enter URLs:")
no_of_sidebars = 3
urls = []
file_name = 'all_url_data_vectors'
# Sidebars for URL input
for i in range(no_of_sidebars):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
# Placeholders for query and progress
query_placeholder = st.empty()
user_query = query_placeholder.text_input("Question: ")
query_button = st.button("Submit Query")
progress_placeholder = st.empty()


if query_button: # on button click
    progress_placeholder.text("Work in Progress...")

    # Loading URL Data in form of Text
    url_loader = UnstructuredURLLoader(urls=urls)
    url_data = url_loader.load()

    # Splitting loaded data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ' '],
        chunk_size=1000,
    )
    progress_placeholder.text("Work in Progress: Text Splitting")
    chunked_url_data = text_splitter.split_documents(url_data)

    # Create Embeddings
    embedding_creator = OpenAIEmbeddings()
    progress_placeholder.text("Work in Progress: Creating Embeddings")
    data_vectors = FAISS.from_documents(chunked_url_data, embedding_creator)
    # Save Embeddings
    data_vectors.save_local(file_name)
    
    if os.path.exists(file_name): # check for testing file saving
        progress_placeholder.text("Work in Progress: Loading Results")
        # fetching data vectors
        data_vectors_loaded = FAISS.load_local(file_name, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        # querying LLM
        main_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm_openai, retriever=data_vectors_loaded.as_retriever())
        llm_result = main_chain({'question': user_query})
        st.header('Answer:')
        # fetching and printing LLM's answer
        st.write(llm_result['answer'])
        # getting source(s) of answer from llm
        answer_sources = llm_result.get('sources','') # check for no sources
        if answer_sources:
            answer_sources_list = answer_sources.split('\n')
            st.subheader('Sources:')
            for source in answer_sources_list:
                st.write(source)