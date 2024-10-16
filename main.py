import streamlit as st

from src.utilities import initialize_milvus, extract_text_with_pypdf, emb_text, get_context, get_response_GPT

print("Imported Packages...")

def main():
    st.title("<p style='text-align: center; font-weight: bold;'>SmartAudit</p>", unsafe_allow_html=True)

    # Initialize Milvus and load collection
    collection = initialize_milvus()

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for user question
    if prompt := st.chat_input("Ask something about IFRS documents..."):
        # Append user's query to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Embed the user's query using OpenAI embeddings
        #query_embedding = emb_text(prompt)

        # Search the collection for relevant texts
        retrieved_texts = get_context(prompt, collection)

        # Generate a response using OpenAI's LLM, augmented with retrieved texts
        llm_response = get_response_GPT(retrieved_texts, prompt)

        # Append the LLM-generated response to chat history
        st.session_state.messages.append({"role": "assistant", "content": llm_response})

        # Display the response
        with st.chat_message("assistant"):
            st.markdown(llm_response)

if __name__ == "__main__":
    main()
