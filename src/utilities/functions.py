
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
from openai import OpenAI
import numpy as np
import time
from tqdm.auto import tqdm
import tiktoken
import concurrent.futures
import streamlit as st

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import faiss
# from pymilvus import MilvusClient



load_dotenv()
API_KEY=os.getenv('OPEN_API_KEY')
# zilli_api_key=os.getenv('zilliAPI_KEY')
# zilliuri = os.getenv('connectionUri')

client = OpenAI(api_key=API_KEY)


SYSTEM_PROMPT = """
    Human: You are a Financial assistant called SmartAudit knowledgeable with the IFRS and GAAP standard for auditing reports.
    You are able to find answers to the questions from the contextual passage snippets provided and you are free to browse the internet for additional information on the IFRS and GAAP standard to augment your response.
    Try to always limit the conversation to be around financial topic related to auditing.
    """


@st.dialog("Disclaimer: ")
def show_modal():
    
    st.write("Welcome! Your data privacy is our priority. Any information you provide is used only to generate responses during your interaction with us. We do not store or share your data.")

def extract_text_with_pypdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def query_pdf_GPT(pdf_content, question):

    SYSTEM_PROMPT = """
    Human: You are a Financial assistant called SmartAudit knowledgeable with the IFRS and GAAP standard for auditing reports.
    You are able to find answers to the questions from the contextual passage snippets provided and you are free to browse the internt for additional information on the IFRS and GAAP standard to augment your response.
    Try to always limit the conversation to be around financial topic related to auditing.
    """
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {pdf_content}
    </context>
    <question>
    {question}
    </question>
    """
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
    max_tokens=6000,
    temperature=0.5,
    stream=True
    )
    
    return response

    # return response.choices[0].message.content

def stream_response(response):
    output = ""
    for message in response:
        token = message.choices[0].delta.content
        if token:
            # print(token, end="")
            output += token
            yield f"""{token}"""
            # Add a delay between chunks to reduce stream speed
            time.sleep(0.05) 


def calculate_token_usage(prompt, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    prompt_tokens = encoding.encode(prompt)
    return len(prompt_tokens)

# cost = (total_tokens / 1_000_000) * 3.00


def extract_text_by_page(pdf_path):
    """
    Function to run text extraction by pages:
    """
    text_by_page = []
    pdf_reader = PdfReader(pdf_path)
    total_pages = len(pdf_reader.pages)
    
    # Initialize the progress bar
    progress_bar = st.progress(0)
    
    for page_number in range(total_pages):
        page = pdf_reader.pages[page_number]
        page_text = page.extract_text()  
        text_by_page.append(page_text)
        
        progress_bar.progress((page_number + 1) / total_pages)  # Update progress bar
    
    progress_bar.progress(1.0)  # Set to 100% after extraction is complete
    return text_by_page

def extract_textfile_into_pages(uploaded_file, lines_per_page=30):
    
    text_content = uploaded_file.read().decode("utf-8")
    lines = text_content.splitlines()  # Split the text into lines
    pages = []
    for i in range(0, len(lines), lines_per_page):
        page_content = "\n".join(lines[i:i + lines_per_page])  
        pages.append(page_content)
    return pages


def query_pdf_GPT_batch(pdf_content, question):
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {pdf_content}
    </context>
    <question>
    {question}
    </question>
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        max_tokens=6000,
        temperature=0.5,
    )
    return response.choices[0].message.content if response.choices else ""

def get_responses_concurrently(pdf_pages, question):
    responses = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        responses = list(executor.map(lambda page: query_pdf_GPT_batch(page, question), pdf_pages))
    return responses


def summarize_responses(responses, question):
    combined_content = "\n".join(responses)
    summary_prompt = f"""
    Please provide a concise summary based on the following information extracted from multiple pages, focusing on key auditing-related insights relevant to IFRS and GAAP standards.
    
    <information>
    {combined_content}
    </information>
    <question>
    {question}
    </question>
    """
    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": summary_prompt},
        ],
        max_tokens=6000,
        temperature=0.5,
        stream=True
    )
    
    return summary_response



#############################################################
## ---- Functions for RAG architecture ----------------------

# st.cache_resource
# def initialize_milvus():

#     milvus_client = MilvusClient(uri="https://in03-cdcb64273091f72.serverless.gcp-us-west1.cloud.zilliz.com",
#                              token=zilli_api_key)
#     collection_name = "demo_rag_collection"

#     milvus_client.load_collection(
#     collection_name=collection_name,
#     replica_number=1 
#                                 )
#     return milvus_client

# #function to chunk text :
# st.cache_data
# def chunk_text(text, chunk_size=5000, chunk_overlap=250):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return splitter.split_text(text)

# # embedding the chunks;
# st.cache_data
# def emb_text(text):
#     return (
#         client.embeddings.create(input=text, model="text-embedding-3-small")
#         .data[0]
#         .embedding
#     )
# st.cache_data
# def embed_chunks(chunks):
#     output = []
#     for chunk in tqdm(chunks, total=len(chunks)):
#         output.append(emb_text(chunk))
#     return output    

# st.cache_data
# def create_faiss_index(embeddings):
#     dim = 1536  
#     print(f"==== creating the Faiss embeddings =======")
#     index = faiss.IndexFlatL2(dim)  # Create a FAISS index with L2 distance metric
#     index.add(embeddings)  
#     return index


# st.cache_data
# def get_context(question, milvus_client, collection_name='demo_rag_collection'):

#     search_res = milvus_client.search(
#     collection_name=collection_name,
#     data=[
#         emb_text(question)
#     ],  # Using the `emb_text` function to convert the question to an embedding vector
#     limit=1,  # here we'd be returning the top 1 result to save query time, much faith in our language model yh? (lol)
#     search_params={"metric_type": "COSINE", "params": {}}, 
#     output_fields=["text"],  
#                                     )
    
#     retrieved_lines_with_distances = [
#     (res["entity"]["text"], res["distance"]) for res in search_res[0]
#                                     ]
#     #print(f" here's the retrieved distances: {retrieved_lines_with_distances}")

#     context = "\n".join(
#     [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
#                         )
    
#     return context

# def get_response_GPT(retrieved_texts, question):

#     SYSTEM_PROMPT = """
#     Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
#     """
#     USER_PROMPT = f"""
#     Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
#     <context>
#     {retrieved_texts}
#     </context>
#     <question>
#     {question}
#     </question>
#     """
#     response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": USER_PROMPT},
#     ],
#     max_tokens=1500,
#     temperature=0.5
#     )

#     return response.choices[0].message.content


# def summarize_response(combined_text):
#     system_prompt = "You are an Auditor and you have been presented with different pieces of information, you are very professional and pay attention to details"

#     user_prompt = f"Summarize the following texts that are outputs from parts of a financial pdf that an anomaly is to be detected or not and make it coherent:\n\n{combined_text}"
#     response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt},
#     ],
#     max_tokens=1500,
#     temperature=0.5
#     )

#     return response.choices[0].message.content
