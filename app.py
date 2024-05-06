import streamlit as st
import os
from PyPDF2 import PdfReader
from transformers import pipeline
import torch
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
import tensorflow


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
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def count_words_in_pdf():
    # upload the file
    pdf_docs =  st.file_uploader("Upload a PDF file", type=["pdf"])
    # Initialize word count
    word_count = 0

    # Iterate through each page of the PDF
    for page_number in range(len(uploaded_file)):
        # Get the text of the page
        page_text = uploaded_file[page_number].get_text()
        
        # Split the text into words and update the word count
        word_count += len(page_text.split())

    # Close the PDF document
    uploaded_file.close()

    return word_count

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
    Write a summary of the following pdf_docs:
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
    st.write("Reply: ", response["output_text"])

    

def main():
    st.image("https://images.app.goo.gl/v6naCB9M3Kh29dKC9", width=100)
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

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
                st.success("Done")
    
    if user_question!='Summary' and user_question!='summary':
        user_input(user_question)

    
    elif user_question=='Summary' or user_question=='summary':
        with st.spinner("Summarizing text..."):
            # if count_words_in_pdf() > 3000:
            summary_1 = summarize_long_pdf(pdf_text)
            # else:
            summary_2 = summarize_short_pdf(pdf_text)

    st.header("Summary_long:")
    st.write(summary_1)
    st.header("Summary_short:")
    st.write(summary_2)

if __name__ == "__main__":
    main()