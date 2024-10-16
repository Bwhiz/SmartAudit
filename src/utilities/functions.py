from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
from openai import OpenAI
from pymilvus import MilvusClient
import numpy as np
from pymilvus import Collection, connections


load_dotenv()
API_KEY=os.getenv('OPEN_API_KEY')
zilli_api_key=os.getenv('zilliAPI_KEY')
zilliuri = os.getenv('connectionUri')


client = OpenAI(api_key=API_KEY)

def initialize_milvus():

    milvus_client = MilvusClient(uri="https://in03-cdcb64273091f72.serverless.gcp-us-west1.cloud.zilliz.com",
                             token=zilli_api_key)
    collection_name = "demo_rag_collection"

    milvus_client.load_collection(
    collection_name=collection_name,
    replica_number=1 
                                )
    return milvus_client

# def initialize_milvus():
#     connections.connect(
#         alias="default", 
#         uri=zilliuri,
#         token=zilli_api_key
#     )
#     collection_name = "demo_rag_collection"
#     collection = Collection(collection_name)
#     collection.load()  # Load collection into memory
#     return collection


def extract_text_with_pypdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# embedding the chunks;
def emb_text(text):
    return (
        client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

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