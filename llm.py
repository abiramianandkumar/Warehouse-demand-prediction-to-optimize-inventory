import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Function to process the response from the LLM
def process_response(response):
    # Clean up the response to ensure it only includes the answer
    return response.strip().split("Answer:")[1].strip() if "Answer:" in response else response.strip()

# Function to initialize and query the LLM
def query_llm(query):
    try:
        # Initialize the LLM with the HuggingFace model
        print('Initializing the LLM...')
        llm = HuggingFaceEndpoint(
            repo_id='mistralai/Mistral-7B-Instruct-v0.3',  # Replace with your desired model
            temperature=0.7,
            max_new_tokens=150
        )
        print('LLM Initialized.')

        # Create the prompt template
        print('Setting up prompt...')
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful medical assistant. Provide a clear and concise answer to the user's specific question. Do not include any extra information, context, or additional questions.

        Question: {input}
        Answer:
        """)
        print('Prompt created.')

        # Format the prompt with the user's input
        full_prompt = prompt.format(input=query)
        print('Querying the LLM...')
        response = llm.invoke(full_prompt)
        print('Response received:', response)

        # Process the response before returning
        return process_response(response)
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Something went wrong. Please try again!"

# Streamlit UI
st.title("Medical Assistant Chatbot")

# Input text box for the user's query
user_query = st.text_input("Ask a question:")

# Button to submit the question
if st.button("Ask"):
    if user_query:
        # Query the LLM and display the result
        answer = query_llm(user_query)
        st.write(f"🤖 Assistant: {answer}")
    else:
        st.write("Please enter a question.")
