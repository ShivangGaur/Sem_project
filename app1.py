import streamlit as st
import os
from PyPDF2 import PdfReader
# from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.chains import LLMChain
from langchain import PromptTemplate

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def read_pdf(uploaded_file):
    # Create the temporary directory if it doesn't exist
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Save the uploaded file to the temporary location
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read the PDF file
    text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    
    # Remove the temporary file
    os.remove(file_path)

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context then do the semantic search and analyze the pdf as a scientific researcher and give something related to that question from the provided context,if the question is completely random just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# def count_words_in_pdf():
#     # upload the file
#     pdf_docs =  st.file_uploader("Upload a PDF file", type=["pdf"])
#     # Initialize word count
#     word_count = 0

#     # Iterate through each page of the PDF
#     for page_number in range(len(uploaded_file)):
#         # Get the text of the page
#         page_text = uploaded_file[page_number].get_text()
        
#         # Split the text into words and update the word count
#         word_count += len(page_text.split())

#     # Close the PDF document
#     uploaded_file.close()

#     return word_count

def summarize_long_pdf(text):
    llm = ChatGoogleGenerativeAI(temperature=0.3, model="gemini-pro")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    chunks = text_splitter.create_documents([text])

    chain = load_summarize_chain(
        llm,
        chain_type='map_reduce',
        verbose=False
    )
    summary = chain.run(chunks)
    return summary

def summarize_short_pdf(text):
    generic_template = '''
    Write a detailed summary of the following pdf_docs:
    Speech : `{pdf_docs}`
    .
    '''
    prompt = PromptTemplate(input_variables=['pdf_docs'], template=generic_template)
    complete_prompt = prompt.format(pdf_docs=text)

    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    summary = llm_chain.run({'pdf_docs': text})
    return summary

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    answer =  response["output_text"]
    return answer

def main():
    st.title("Chat Messaging Application")
    st.markdown("""
    <style>
    .st-eb {
        background-color: #f0f0f0 !important;
        border-radius: 15px !important;
        margin-bottom: 20px !important;
        max-height: 50% !important;
    }
    .st-ec {
        border: none !important;
        max-height: 50% !important;s     
                
    }
    .user-message {
        text-align: right !important;
        display: inline-block;
        margin-left: 15%;
        background-color: #DCF8C6 !important;
        border: 1px solid #b2e0a1;
        border-radius: 15px 15px 0px 15px !important;
        padding: 10px;
        margin-bottom: 15px !important;
        font-family: Arial, sans-serif;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        word-wrap: break-word;
        float: right;
    }
    .bot-message {
        text-align: left !important;
        display: inline-block;
        margin-right: 15%;
        background-color: #D9E6F5 !important;
        border: 1px solid #a8c4e4;
        border-radius: 15px 15px 15px 0px !important;
        margin-bottom: 15px !important;
        padding: 10px;
        font-family: Arial, sans-serif;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        word-wrap: break-word;
        float: left;
    }
    .message-container {
        max-width: auto;
        margin-left: auto;
        margin-right: auto;
        max-height: 50%;
        overflow-y: auto;
    }
    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    bot_response = ""

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    user_question = st.text_area("Ask a Question from the PDF Files")
    # Main content area for displaying messages

    if "pdf_file" not in st.session_state:
        st.session_state.pdf_file = None
    
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""

    with st.sidebar:
        st.title("Menu:")
        pdf_docs =  st.file_uploader("Upload a PDF file", type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = read_pdf(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                pdf_reader = PdfReader(pdf_docs)
                pdf_text=""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pdf_text += page_text
                st.session_state.pdf_text = pdf_text
                st.success("Done")
                
    # Display messages
    # messages = [
    #     {"sender": "User", "message": user_question},
    # ]
    
    if st.button("Send"):
        if user_question:
            st.session_state.messages.append({"sender": "User", "text": user_question})

            if user_question not in ['summary', 'summarize']:
                answer = user_input(user_question)
            else:
                with st.spinner("Summarizing text..."):
                    answer_long = summarize_long_pdf(st.session_state.pdf_text)
                    answer_short = summarize_short_pdf(st.session_state.pdf_text)
                    answer = f"Long Summary:\n{answer_long}\n\nShort Summary:\n{answer_short}"

            st.session_state.messages.append({"sender": "Bot_response", "text": answer})

    st.markdown('<div class="message-container">', unsafe_allow_html=True)
    for i, msg in enumerate(st.session_state.messages):
        #message(msg["text"], is_user=(msg["sender"] == "User"), key=f"msg_{i}")
        css_class = "user-message" if msg["sender"] == "User" else "bot-message"
        st.markdown(
            f'<div class="{css_class}">{msg["text"]}</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)
            


if __name__ == "__main__":
    main()
#12 feet ladder