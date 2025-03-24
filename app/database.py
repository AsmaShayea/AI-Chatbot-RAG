from pymongo import MongoClient
from flask_pymongo import PyMongo
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus

def get_database():
    username = "admin"
    password = "M0nG0@dm1n!2025$"
    encoded_password = quote_plus(password)
    connection_string = f"mongodb://{username}:{encoded_password}@16.16.57.62:27017/admin?authSource=admin"
    
    client = MongoClient(connection_string, server_api=ServerApi('1'))
    db = client["chatbot"]
    # client = MongoClient("mongodb://localhost:27017/", server_api=ServerApi('1'))
    # db = client["chatbot_saas"]
    return db

db = get_database()
