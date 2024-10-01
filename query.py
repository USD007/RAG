import streamlit as st
import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from embedding import get_embedding_function

CHROMA_PATH = "chroma"

# Define the template for the chat prompt
template = """
You are a document assistant designed to help users answer questions related to documents. 
Please use the information only and only from the provided context to answer the question.
if there is no relevant info from the context please say I have no idea on this 
Context:
{context}

---

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOllama(model="llama3")  # Replace with your specific model name

def main():
    st.title("RAG Application")

    # User input for question
    user_input = st.text_input("Ask a question about your document:")


    # Timer to track response time
    if "response_time" not in st.session_state:
        st.session_state.response_time = 0


    if st.button("Send"):
        if user_input:
            # Query the vector database for relevant context
            start_time = time.time()
            embedding_function = get_embedding_function()
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            results = db.similarity_search_with_score(user_input, k=2)

            # Extract context from the search results
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
            prompt_input = prompt.format(context=context_text, question=user_input)

            # Invoke the model with the formatted prompt
            response_text = model.invoke(prompt_input)
            response_msg = response_text.content

            # Stop the timer
            end_time = time.time()
            st.session_state.response_time = end_time - start_time

            # Display the response
            st.write("Response:", response_msg)
        else:
            st.write("Please enter a question.")
     # Display the response time in the sidebar
    st.sidebar.write(f"Response Time: {st.session_state.response_time:.2f} seconds")


if __name__ == "__main__":
    main()
