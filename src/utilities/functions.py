from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
import numpy as np

load_dotenv()
API_KEY=os.getenv('OPEN_API_KEY')
zilli_api_key=os.getenv('zilliAPI_KEY')

client = OpenAI(api_key=API_KEY)

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