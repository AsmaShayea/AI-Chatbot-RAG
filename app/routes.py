import os
import uuid
from flask import Blueprint, request, jsonify
from app.database import db, get_database
from app.chatbot import create_vectorstore, get_chatbot_response
from werkzeug.utils import secure_filename
from flask import request
from flask_cors import CORS 

api_bp = Blueprint("api", __name__)    
CORS(api_bp, origins="*")  

db = get_database()

# Uploads folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

### **1. Home Route**
@api_bp.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Flask Chatbot"}), 200


### **2. Create Chatbot (Upload Documents)**
@api_bp.route("/create_chatbot", methods=["POST"])
def create_chatbot():
    """Allows users to create a chatbot with document uploads."""
    bot_name = request.form.get("bot_name")
    urls = request.form.getlist("urls")  # ✅ Fixing URL list retrieval
    file = request.files.get("file")

    if not bot_name:
        return jsonify({"error": "Missing chatbot name"}), 400

    chatbot_id = str(uuid.uuid4())

    if file:
        filename = f"{chatbot_id}__{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
    else:
        filepath = None

    # Log for debugging
    print(f"Bot Name: {bot_name}")
    print(f"URLs: {urls}")
    print(f"File: {file.filename if file else 'No file uploaded'}")

    # Create vectorstore with uploaded documents or URLs
    # vectorstore_path, error = create_vectorstore(chatbot_id, urls=urls)
    vectorstore_path, error = create_vectorstore(chatbot_id, urls=urls, file_path=filepath)

    if error:
        return jsonify({"error": error}), 400

    # Save chatbot metadata in MongoDB
    db.chatbots.insert_one({
        "chatbot_id": chatbot_id,
        "bot_name": bot_name,
        "vectorstore_path": vectorstore_path,
        "chat_history": []
    })

    return jsonify({"message": "Chatbot created successfully!", "chatbot_id": chatbot_id}), 201


### **3. Chat with Chatbot**
@api_bp.route("/chat/<chatbot_id>", methods=["POST"])
def chat(chatbot_id):
    print(f"Received chatbot_id: {chatbot_id}")
    data = request.json
    user_question = data.get("question")
    model_name = data.get("model_name")

    if not user_question:
        return jsonify({"error": "Missing 'question' field"}), 400

    return get_chatbot_response(chatbot_id, user_question, model_name)



@api_bp.route('/chat/<chatbot_id>/history', methods=['GET'])
def get_chat_history(chatbot_id):
    model_name = request.args.get("model_name")
    if not model_name:
        return jsonify({"error": "Missing model_name"}), 400
    chatbot_data = db.chatbots.find_one({"chatbot_id": chatbot_id})
    if not chatbot_data:
        return jsonify({"error": "Chatbot not found."}), 404

    model_history_title = model_name + "_chat_history"
    chat_history = chatbot_data.get(model_history_title, [])
    return jsonify(chat_history), 200

