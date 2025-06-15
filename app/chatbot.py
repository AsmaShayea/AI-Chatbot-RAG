import os
import re
from datetime import datetime
from flask import jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader, CSVLoader
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from app.database import db
import warnings
from langdetect import detect

# Suppress warnings
warnings.filterwarnings('ignore')

# Environment and configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables!")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in environment variables!")

VECTORSTORE_DIR = "./chroma_db"

# Model configuration
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

# Initialize the embedding model globally
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


SYSTEM_PROMPT = """
You are a helpful AI assistant.

Your job is to answer user questions using only information from the uploaded documents or provided website(s).

**Rules:**
- You must answer in {detected_lang} language.
- Be extremely concise — respond very briefly unless more detail is necessary.
- Keep answers as short as possible.
- If the user asks a question, answer that specific question clearly and directly using only the uploaded content.
- If the user just greets you or asks what you do (e.g., "hello", "who are you?"), then briefly describe what the uploaded content is about in 2–3 concise sentences.
- Do NOT say “in your file/website” or use generic phrases — answer naturally as if you're the source.
- **Never mention the source or say phrases like “from the website,” “from the file,” or “Reiterated from.” Just answer directly.**

**If the user's question is a greeting or a general inquiry (such as 'who are you?' or 'what do you know?'):**
1. Briefly introduce yourself by stating you are an assistant for answering questions about [insert a **short summary or title of the uploaded files or website(s)**]. Example: "I'm your assistant for answering questions about [short overview here]."
2. Briefly say what kind of information you can provide, based on the specific content of the uploaded files or website(s).

**For all other questions:**
- ONLY answer if the information exists in the documents or website(s). Always provide a short, clear answer.
- If you cannot find an answer, reply: "Sorry, I couldn't find relevant information in the provided materials."
"""

# Helper functions
def get_llm(model_name):
    config = MODEL_MAP.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name in LLM_INSTANCES:
        return LLM_INSTANCES[model_name]

    provider = config["provider"]

    if provider in ["openai", "deepseek"]:
        api_key = os.getenv(config.get("api_key_env"))
        if not api_key:
            raise ValueError(f"Missing {config.get('api_key_env')} in env!")
        llm = ChatOpenAI(
            model=config["model"],
            api_key=api_key,
            base_url=config.get("base_url", None)
        )
    elif provider == "huggingface":
        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            raise ValueError("Missing HUGGINGFACE_HUB_TOKEN in environment!")
        llm = ChatHuggingFace(
            repo_id=config["model"],
            huggingfacehub_api_token=hf_token
        )
    elif provider == "anthropic":
        api_key = os.getenv(config.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY in env!")
        llm = ChatAnthropic(
            model=config["model"],
            api_key=api_key
        )
    elif provider == "google":
        api_key = os.getenv(config.get("api_key_env", "GOOGLE_API_KEY"))
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in env!")
        llm = ChatGoogleGenerativeAI(
            model=config["model"],
            api_key=api_key
        )
    else:
        raise ValueError(f"Provider '{provider}' not implemented.")

    LLM_INSTANCES[model_name] = llm
    return llm

def load_documents(urls=None, file_path=None):
    documents = []

    if file_path:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path=file_path)
            else:
                loader = None

            if loader:
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading file: {file_path} -> {e}")

    if urls:
        try:
            loader = WebBaseLoader(urls)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading URLs: {e}")

    return documents

def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    return text_splitter.split_documents(data)

def create_vectorstore(chatbot_id, urls=None, file_path=None):
    documents = load_documents(urls, file_path)
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
        return None

def build_prompt(question, context):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "{context}\nUser: {question}")
    ])
    return prompt.format(context=context, question=question)

def get_chatbot_response(chatbot_id, question, model_name):
    llm = get_llm(model_name)
    chatbot_data = db.chatbots.find_one({"chatbot_id": chatbot_id})
    if not chatbot_data:
        return jsonify({"error": "Chatbot not found."}), 404

    model_history_title = model_name + "_chat_history"
    chat_history = chatbot_data.get(model_history_title, [])
    context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

    try:
        retriever_obj = retriever(chatbot_id)
        if not retriever_obj:
            return jsonify({"error": "Retriever not found."}), 500
        
        try:
            user_lang = detect(question)
        except:
            user_lang = "en"

        filled_prompt = SYSTEM_PROMPT.format(detected_lang="Arabic" if user_lang == "ar" else "English")
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": ChatPromptTemplate.from_messages([
                    ("system", filled_prompt),
                    ("user", "{context}\nUser: {question}")
                ])
            }
        )

        history_aware_input = f"{context}\nUser: {question}"
        response = qa({"query": history_aware_input})



        # Save user message
        user_message = {
            "role": "user",
            "content": question,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        chat_history.append(user_message)


        answer = "Sorry, I couldn't find relevant information in the provided documents."
        if response['result'] and "I don't know" not in response['result']:
            answer = response['result']

        bot_message = {
            "role": "ai",
            "content": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        chat_history.append(bot_message)

        db.chatbots.update_one(
            {"chatbot_id": chatbot_id},
            {"$set": {model_history_title: chat_history}}
        )

        return answer

    except Exception as e:
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500