import streamlit as st
from src.utilities import (initialize_milvus, extract_text_with_pypdf, emb_text, get_context, query_pdf_GPT,
                           get_response_GPT, chunk_text, query_llm_with_chunk, summarize_response, 
                           stream_response, embed_chunks, create_faiss_index, calculate_token_usage)
import numpy as np
from prometheus_client import CollectorRegistry, Counter, Summary, start_http_server

st.set_page_config(
    page_title="SmartAudit",
    page_icon="🗯"
)
if "registry" not in st.session_state:
    registry = CollectorRegistry()
    st.session_state.registry = registry

    # Start Prometheus server only once, using custom registry
    start_http_server(8050, registry=st.session_state.registry)

    # Register metrics in the custom registry
    st.session_state.token_counter = Counter('openai_tokens_total', 'Total number of tokens used', registry=registry)
    st.session_state.input_cost_summary = Summary('openai_cost_dollars', 'Total API cost for Prompts in dollars', registry=registry)
    st.session_state.completion_cost_summary = Summary('openai_completion_cost_dollars', 'Total API cost for Output Tokens in dollars', registry=registry)

# except Exception as e:
#     print(f"An error with details {e} just occured here")

token_counter = st.session_state.token_counter
input_cost_summary = st.session_state.input_cost_summary
completion_cost_summary = st.session_state.completion_cost_summary

print("Imported Packages...")

def main():
    st.markdown("<h1 style='text-align: center;'>SmartAudit</h1>", unsafe_allow_html=True)

    uploaded_pdf = st.sidebar.file_uploader("Upload your PDF document", type="pdf")
    if uploaded_pdf is not None:
        # Extract text from the uploaded PDF
        text = extract_text_with_pypdf(uploaded_pdf)
        # pdf_chunks = chunk_text(text)
        
        # Store embeddings in session state
        if 'pdf_embeddings' not in st.session_state:
            st.session_state['pdf_embeddings'] = []
        st.session_state['pdf_embeddings'].append(text)
    else:
        st.session_state['pdf_embeddings'] = []
        

    # Initialize Milvus and load collection
    #collection = initialize_milvus()

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for user question
    if prompt := st.chat_input("Ask something about the document..."):
        # Append user's query to chat history and display immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

        # Check if the PDF content exists
        if len(st.session_state['pdf_embeddings']) != 0:
            # Get the last PDF content for querying
            pdf_content = st.session_state['pdf_embeddings'][-1]       
        else:
            pdf_content = ""

        response_generator = query_pdf_GPT(pdf_content, prompt)
        
        full_response = ""
        for token in stream_response(response_generator):
            full_response += token
            message_placeholder.markdown(full_response)  # Update placeholder with new content

        prompt_token = calculate_token_usage(prompt + pdf_content)
        completion_token = calculate_token_usage(full_response)

        total_tokens_used = prompt_token + completion_token

        input_cost = (total_tokens_used / 1_000_000) * 2.50
        output_cost = (completion_token / 1_000_000) * 10.00

        total_cost = input_cost + output_cost

        token_counter.inc(total_tokens_used) 
        input_cost_summary.observe(total_cost)
        completion_cost_summary.observe(output_cost)

        # Append the final LLM-generated response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    
    main()
