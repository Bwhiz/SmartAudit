import streamlit as st
from src.utilities import (query_pdf_GPT, stream_response,  calculate_token_usage, extract_text_by_page, 
                           get_responses_concurrently, summarize_responses, show_modal)

from prometheus_client import CollectorRegistry, Counter, Summary, start_http_server

st.set_page_config(
    page_title="SmartAudit",
    page_icon="🗯"
)

# try:
#     if "registry" not in st.session_state:
#         registry = CollectorRegistry()
#         st.session_state.registry = registry

#         # Start Prometheus server only once, using custom registry
#         start_http_server(8050, registry=st.session_state.registry)

#         # Register metrics in the custom registry
#         st.session_state.token_counter = Counter('openai_tokens_total', 'Total number of tokens used', registry=registry)
#         st.session_state.input_cost_summary = Summary('openai_cost_dollars', 'Total API cost for Prompts in dollars', registry=registry)
#         st.session_state.completion_cost_summary = Summary('openai_completion_cost_dollars', 'Total API cost for Output Tokens in dollars', registry=registry)

#     token_counter = st.session_state.token_counter
#     input_cost_summary = st.session_state.input_cost_summary
#     completion_cost_summary = st.session_state.completion_cost_summary

# except (OSError, KeyError, AttributeError) as e:
#     print(f"=== Another User is on the application ===\n Error ==> {e}")
#     pass


print("Imported Packages...")

if "registry" not in st.session_state:
    registry = CollectorRegistry()
    st.session_state.registry = registry
    st.session_state['modal_shown'] = False    

    # Show the modal if it hasn't been shown yet
    if not st.session_state['modal_shown']:
        show_modal()

def main():
    st.markdown("<h1 style='text-align: center;'>SmartAudit</h1>", unsafe_allow_html=True)

    if 'pdf_text' not in st.session_state:
        st.session_state['pdf_text'] = [] 

    if 'uploaded_filename' not in st.session_state:
        st.session_state['uploaded_filename'] = None

    uploaded_pdf = st.sidebar.file_uploader("Upload your PDF document", type="pdf")
    
    if uploaded_pdf is not None:
        if st.session_state['uploaded_filename'] != uploaded_pdf.name:
            with st.spinner("Processing uploaded Document ..."):
                # Extract text and save to session state
                st.session_state['pdf_text'] = extract_text_by_page(uploaded_pdf)
                st.session_state['uploaded_filename'] = uploaded_pdf.name
                #st.session_state['pdf_embeddings'] = st.session_state['pdf_text']
    else:
        st.session_state['pdf_text'] = []


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
            message_placeholder.markdown("Scanning through Doc 🔎📄...")


        # Check if the PDF content exists
        if len(st.session_state['pdf_text']) != 0:
            # Get the last PDF content for querying
            pdf_content = st.session_state['pdf_text']     
        else:
            pdf_content = ""

        if pdf_content != "" :
            response_generator = get_responses_concurrently(pdf_content, prompt)

            final_summary = summarize_responses(response_generator, prompt)
        else:
            final_summary = query_pdf_GPT(pdf_content, prompt)

        full_response = ""
        for token in stream_response(final_summary):
            full_response += token
            message_placeholder.markdown(full_response)  # Update placeholder with new content

        # prompt_token = calculate_token_usage(prompt + "\n".join(pdf_content))
        # completion_token = calculate_token_usage(full_response)

        # total_tokens_used = prompt_token + completion_token

        # input_cost = (total_tokens_used / 1_000_000) * 2.50
        # output_cost = (completion_token / 1_000_000) * 10.00

        # total_cost = input_cost + output_cost

        # try:
        #     token_counter.inc(total_tokens_used) 
        #     input_cost_summary.observe(total_cost)
        #     completion_cost_summary.observe(output_cost)
        # except Exception as e:
        #     print("Error encountered trying to access unintialized session states")
        #     pass

        # Append the final LLM-generated response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    
    main()
