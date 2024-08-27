import streamlit as st
from chatBot import build_llm, build_retrieval_qa, qa_prompt, vectorstore

def main():
    st.set_page_config(page_title="Q&A Chatbot", page_icon="💬")
    st.header("Q&A Chatbot 💬")
    
    # Store the conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    
    # Store the chat history
    if 'chatHistory' not in st.session_state:
        st.session_state.chatHistory = []
    
    # User question input
    user_question = st.text_input("Ask a Question")
    
    if user_question:
        # Add user's question to the chat history
        st.session_state.chatHistory.append(f"You: {user_question}")
        
        # Initialize conversation if not already done
        if st.session_state.conversation is None:
            llm = build_llm()
            st.session_state.conversation = build_retrieval_qa(llm, qa_prompt, vectorstore)
        
        # Get the bot's response
        result = st.session_state.conversation({"query": user_question})
        st.session_state.chatHistory.append(f"Bot: {result['result']}")
        
        # Print source documents if available
        if 'source_documents' in result and result['source_documents']:
            for doc in result['source_documents']:
                source_path = doc.metadata['source'].replace("\\", "/")
                source_info = f"Source: {source_path.split('/')[1]} (Page {doc.metadata.get('page', 'Unknown')})"
                st.session_state.chatHistory.append(source_info)
    
    # Display the chat history
    for message in st.session_state.chatHistory:
        st.write(message)

if __name__ == "__main__":
    main()
