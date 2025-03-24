from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader, CSVLoader
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from flask import jsonify
from app.database import db
from datetime import datetime
import os
import re
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

os.environ["USER_AGENT"] = "ChatbotRag/1.0"

# Environment and configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables!")

llm = None

def get_llm():
    global llm
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    return llm

# llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
VECTORSTORE_DIR = "./chroma_db"

# Initialize the embedding model once globally
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## Document Loader
def load_documents(urls: list = None):
    """Load documents from a folder or URLs and return a list of Document objects."""
    documents = []
    folder_path = 'uploads'
    # Load documents from files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            if filename.endswith('.pdf'):
                print(f"Loading PDF file: {filename}")
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
            elif filename.endswith('.docx'):
                print(f"Loading Word file: {filename}")
                loader = Docx2txtLoader(file_path)
                loaded_docs = loader.load()
            elif filename.endswith('.csv'):
                print(f"Loading CSV file: {filename}")
                loader = CSVLoader(file_path=file_path)
                loaded_docs = loader.load()
            else:
                print(f"Unsupported file type: {filename}")
                continue

            documents.extend(loaded_docs)

        except Exception as e:
            print(f"Error loading file {filename}: {str(e)}")

    # Load documents from web URLs
    if urls:
        try:
            print(f"Loading from URLs: {urls}")
            loader = WebBaseLoader(urls)
            web_docs = loader.load()
            documents.extend(web_docs)
        except Exception as e:
            print(f"Error loading URLs: {str(e)}")

    print(f"Loaded {len(documents)} documents from folder and URLs.")
    return documents


## Text Splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks


## Vector DB Creation
def create_vectorstore(chatbot_id, urls):
    documents = load_documents(urls)
    doc_chunks = text_splitter(documents)
    vectorstore_path = os.path.join(VECTORSTORE_DIR, f"chatbot_{chatbot_id}")

    try:
        vectorstore = Chroma.from_documents(
            collection_name=f"chatbot_{chatbot_id}",
            documents=doc_chunks,
            embedding=embedding_model,
            persist_directory=vectorstore_path
        )
        print(f"Vectorstore created at {vectorstore_path}")
        return vectorstore_path, None
    except Exception as e:
        print(f"Error creating vectorstore: {str(e)}")
        return None, str(e)


vectordb_cache = {}

def retriever(chatbot_id):
    if chatbot_id in vectordb_cache:
        return vectordb_cache[chatbot_id]

    vectorstore_path = os.path.join(VECTORSTORE_DIR, f"chatbot_{chatbot_id}")

    try:
        vectordb = Chroma(
            collection_name=f"chatbot_{chatbot_id}",
            persist_directory=vectorstore_path,
            embedding_function=embedding_model
        )
        vectordb_cache[chatbot_id] = vectordb.as_retriever()
        return vectordb_cache[chatbot_id]
    except Exception as e:
        print(f"Error creating retriever: {str(e)}")
        return None



## QA Chain
def get_chatbot_response(chatbot_id, question):

    llm = get_llm()
    chatbot_data = db.chatbots.find_one({"chatbot_id": chatbot_id})
    if not chatbot_data:
        return jsonify({"error": "Chatbot not found."}), 404
    
    # ✅ Load chat history from the database
    chat_history = chatbot_data.get("chat_history", [])

    # Construct the context from chat history (both user and AI messages)
    context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

    try:
        retriever_obj = retriever(chatbot_id)
        if not retriever_obj:
            return jsonify({"error": "Retriever not found."}), 500

        # Create the RetrievalQA pipeline with chat history awareness
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=False
        )

        # ✅ Log user message to the chat history
        user_message = {
            "role": "user",
            "content": question,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        chat_history.append(user_message)

        # Combine chat history with the current question
        history_aware_input = f"{context}\nUser: {question}"

        response = qa({"query": history_aware_input})

        answer = "No information available"
        # Ensure the response contains a valid answer
        if not response['result'] or "I don't know" in response['result']:
            print("No response from QA chain.")
            answer = "No information available"
        else:
            print(f"result: {response['result']}")
            answer = response['result']

        # ✅ Log bot response to the chat history
        answer = re.sub(r'^Ai:\s+', '', answer, flags=re.IGNORECASE)

        bot_message = {
            "role": "ai",
            "content": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        chat_history.append(bot_message)

        # ✅ Save updated chat history to MongoDB
        db.chatbots.update_one(
            {"chatbot_id": chatbot_id},
            {"$set": {"chat_history": chat_history}}
        )

        return answer

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500


