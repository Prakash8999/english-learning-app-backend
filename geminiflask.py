from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import spacy
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from flask_cors import CORS
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
CORS(app, origins="*")

def get_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.05)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain




def extract_names(text):
    
    nlp = spacy.load("en_core_web_sm")
    
    
    doc = nlp(text)
    
    
    names = set()
    
    
    for entity in doc.ents:
        
        if entity.label_ == "PERSON":

            names.add(entity.text)
    
    
    return list(names)



@app.route('/uploadpdftext', methods=['POST'])
def upload_pdftext():
    text = request.json['text']
    print(text)
    text_chunks = get_text_chunks(text)
    get_vector_store(text_chunks)
    names = extract_names(text)
    return jsonify({"message": "Text uploaded successfully", "names": names})

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json['question']
    charname = request.json['charname']
    print(user_question)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    return jsonify({"response": response["output_text"], "Sender" : "Bot", "Charname": charname})

if __name__ == '__main__':
    app.run(debug=True)
