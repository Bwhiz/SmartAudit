import streamlit as st
from src.utilities import initialize_milvus, extract_text_with_pypdf, emb_text, get_context, get_response_GPT

print("Imported Packages...")

def main():
    st.markdown("<h1 style='text-align: center; font-weight: bold;'>SmartAudit</h1>", unsafe_allow_html=True)

    # Initialize Milvus and load collection
    collection = initialize_milvus()
    
    #divide the page into two columns
    left_col, right_col = st.columns([2,1], vertical_alignment="bottom")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with left_col.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
            elif message["role"] == "user": #shift user response to the right
                st.markdown(
                    """
                    <style>
                        .st-emotion-cache-janbn0 {
                            flex-direction: row-reverse;
                            text-align: right;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True,
                    )
                st.markdown(message["content"])
            
    #display file uploader (will take only one file)    
    file_uploaded = right_col.file_uploader("Upload PDF, DOC", accept_multiple_files = False)

    #React to users (first) input
    if prompt := st.chat_input(placeholder="Write a message"):
        # Display user message in the chat messsage container
        with left_col.chat_message("user"): #shift user response to the right
            st.markdown(
                """
                <style>
                    .st-emotion-cache-janbn0 {
                        flex-direction: row-reverse;
                        text-align: right;
                    }
                </style>
                """,
                unsafe_allow_html=True,
                )
            st.markdown(prompt)
        #add user message to chat history
        st.session_state.messages.append({"role":"user", "content": prompt})

    #Input-Processing-Output block    
    def process(prompt, pdf_input):
        
        ''' 
        This function takes in the text and pdf input, processes it and returns a 
        text output to be displayed
        '''
        
        if pdf_input:
            try:
                #extract text from uploaded pdf
                extracted_pdf = extract_text_with_pypdf(pdf_input.read())
                
                #vectorize the pdf, call the prompt and combine both queries
                
                # Embed the user's query using OpenAI embeddings
                query_embedding = emb_text(prompt)
                # Search the collection for relevant texts
                retrieved_texts = get_context(prompt, collection)
                # Generate a response using OpenAI's LLM, augmented with retrieved texts
                llm_response = get_response_GPT(retrieved_texts, prompt)
            except Exception as e:
                llm_response = f"Error. Unable to process response. {e}"
                
        else: 
            try:
                # Embed the user's query using OpenAI embeddings
                query_embedding = emb_text(prompt)
                # Search the collection for relevant texts
                retrieved_texts = get_context(prompt, collection)
                # Generate a response using OpenAI's LLM, augmented with retrieved texts
                llm_response = get_response_GPT(retrieved_texts, prompt)
            except Exception as e:
                llm_response = f"Error. Unable to handle request. {e}"
        return llm_response

    #get the response
    response = process(prompt,file_uploaded)
    
    #display the response
    with left_col.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content":response})


if __name__ == "__main__":
    main()
