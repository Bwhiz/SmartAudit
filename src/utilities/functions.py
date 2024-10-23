
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
from openai import OpenAI
from pymilvus import MilvusClient
import numpy as np
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from tqdm.auto import tqdm
import tiktoken


load_dotenv()
API_KEY=os.getenv('OPEN_API_KEY')
zilli_api_key=os.getenv('zilliAPI_KEY')
zilliuri = os.getenv('connectionUri')


client = OpenAI(api_key=API_KEY)

st.cache_resource
def initialize_milvus():

    milvus_client = MilvusClient(uri="https://in03-cdcb64273091f72.serverless.gcp-us-west1.cloud.zilliz.com",
                             token=zilli_api_key)
    collection_name = "demo_rag_collection"

    milvus_client.load_collection(
    collection_name=collection_name,
    replica_number=1 
                                )
    return milvus_client

def extract_text_with_pypdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

#function to chunk text :
st.cache_data
def chunk_text(text, chunk_size=5000, chunk_overlap=250):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# embedding the chunks;
st.cache_data
def emb_text(text):
    return (
        client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )
st.cache_data
def embed_chunks(chunks):
    output = []
    for chunk in tqdm(chunks, total=len(chunks)):
        output.append(emb_text(chunk))
    return output    

st.cache_data
def create_faiss_index(embeddings):
    dim = 1536  
    print(f"==== creating the Faiss embeddings =======")
    index = faiss.IndexFlatL2(dim)  # Create a FAISS index with L2 distance metric
    index.add(embeddings)  
    return index


st.cache_data
def get_context(question, milvus_client, collection_name='demo_rag_collection'):

    search_res = milvus_client.search(
    collection_name=collection_name,
    data=[
        emb_text(question)
    ],  # Using the `emb_text` function to convert the question to an embedding vector
    limit=1,  # here we'd be returning the top 1 result to save query time, much faith in our language model yh? (lol)
    search_params={"metric_type": "COSINE", "params": {}}, 
    output_fields=["text"],  
                                    )
    
    retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
                                    ]
    #print(f" here's the retrieved distances: {retrieved_lines_with_distances}")

    context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
                        )
    
    return context

def get_response_GPT(retrieved_texts, question):

    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {retrieved_texts}
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
    max_tokens=1500,
    temperature=0.5
    )

    return response.choices[0].message.content


def query_llm_with_chunk(chunk, prompt):
    system_prompt = "You are an Auditor and you have been presented with different pieces of information, you are very professional and pay attention to details"

    #user_prompt = f"Relevant IFRS standards:\n{ifrs_chunks}\n\nPDF chunk:\n{chunk}\n\n{prompt}."
    user_prompt = f"PDF chunk:\n{chunk}\n\n{prompt}."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
        max_tokens=6000,
        temperature=0.5 
    )
    return response.choices[0].message.content

def summarize_response(combined_text):
    system_prompt = "You are an Auditor and you have been presented with different pieces of information, you are very professional and pay attention to details"

    user_prompt = f"Summarize the following texts that are outputs from parts of a financial pdf that an anomaly is to be detected or not and make it coherent:\n\n{combined_text}"
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    max_tokens=1500,
    temperature=0.5
    )

    return response.choices[0].message.content

def query_pdf_GPT(pdf_content, question):

    SYSTEM_PROMPT = """
    Human: You are a Financial assistant called SmartAudit knowledgeable with the IFRS standard for auditing reports.
    You are able to find answers to the questions from the contextual passage snippets provided and you are free to browse the internt for additional information on the IFRS standard to augment your response.
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

prompt = "Your prompt here"
prompt_tokens = calculate_token_usage(prompt)

# cost = (total_tokens / 1_000_000) * 3.00
