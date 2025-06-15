import os
import re
import warnings
from datetime import datetime
from flask import jsonify
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader, CSVLoader
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from app.database import db

warnings.filterwarnings('ignore')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

VECTORSTORE_DIR = "./chroma_db"

MODEL_MAP = {
    "DeepSeek": {
        "provider": "openai",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY"
    },
    "Llama": {
        "provider": "huggingface",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    },
    "GPT": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY"
    },
    "Gemini": {
        "provider": "google",
        "model": "gemini-2.0-flash",
        "api_key_env": "GOOGLE_API_KEY"
    }
}

LLM_INSTANCES = {}
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb_cache = {}


def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def get_llm(model_name):
    config = MODEL_MAP.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name in LLM_INSTANCES:
        return LLM_INSTANCES[model_name]

    provider = config["provider"]

    if provider in ["openai", "deepseek"]:
        api_key = os.getenv(config.get("api_key_env"))
        llm = ChatOpenAI(
            model=config["model"],
            api_key=api_key,
            base_url=config.get("base_url"),
            temperature=0.7  
        )

    elif provider == "huggingface":
        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        llm = ChatHuggingFace(
            repo_id=config["model"],
            huggingfacehub_api_token=hf_token
        )
    elif provider == "anthropic":
        api_key = os.getenv(config.get("api_key_env", "ANTHROPIC_API_KEY"))
        llm = ChatAnthropic(
            model=config["model"],
            api_key=api_key
        )
    elif provider == "google":
        api_key = os.getenv(config.get("api_key_env", "GOOGLE_API_KEY"))
        llm = ChatGoogleGenerativeAI(
            model=config["model"],
            api_key=api_key
        )
    else:
        raise ValueError(f"Provider '{provider}' not implemented.")

    LLM_INSTANCES[model_name] = llm
    return llm


def load_documents(urls=None):
    documents = []
    folder_path = 'uploads'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
            elif filename.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                loaded_docs = loader.load()
            elif filename.endswith('.csv'):
                loader = CSVLoader(file_path=file_path)
                loaded_docs = loader.load()
            else:
                continue
            documents.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading file {filename}: {str(e)}")

    if urls:
        try:
            loader = WebBaseLoader(urls)
            web_docs = loader.load()
            documents.extend(web_docs)
        except Exception as e:
            print(f"Error loading URLs: {str(e)}")

    return documents


def text_splitter(data):
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50).split_documents(data)


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
        return vectorstore_path, None
    except Exception as e:
        return None, str(e)


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
    except Exception:
        return None


def get_chatbot_response(chatbot_id, question, model_name):
    llm = get_llm(model_name)
    chatbot_data = db.chatbots.find_one({"chatbot_id": chatbot_id})
    if not chatbot_data:
        return jsonify({"error": "Chatbot not found."}), 404

    model_history_title = model_name + "_chat_history"
    chat_history = chatbot_data.get(model_history_title, [])
    context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

    user_lang = detect_language(question)
    lang_instruction = {
        "en": "Answer in English only.",
        "ar": "أجب باللغة العربية فقط."
    }.get(user_lang, "Respond in the same language as the question.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""# Role\nYou are a concise and helpful assistant that only uses the provided document and website content to answer questions.\n\n# Behavior Rules\n- Always reply in the same language as the question.\n- {lang_instruction}\n- Avoid generic phrases like 'in your document' or 'from the website'.\n- Answer only if you find relevant info in the sources.\n\n# Instruction\nUse only the relevant content from context to answer. Be specific, avoid repetition, and don't invent answers."""),
        ("user", "# Context\n{context}\n\n# Question\n{question}")
    ])

    try:
        retriever_obj = retriever(chatbot_id)
        if not retriever_obj:
            return jsonify({"error": "Retriever not found."}), 500

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,