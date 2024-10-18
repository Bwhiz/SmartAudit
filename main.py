import streamlit as st
from src.utilities import (initialize_milvus, extract_text_with_pypdf, emb_text, get_context, query_pdf_GPT,
                           get_response_GPT, chunk_text, query_llm_with_chunk, summarize_response, stream_response)

print("Imported Packages...")

st.set_page_config(
    page_title="SmartAudit",
    page_icon="🗯"
)


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

        # Append the final LLM-generated response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
