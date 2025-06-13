from config import MONGO_URI
from pymongo import MongoClient
from flask_pymongo import PyMongo
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
import os

def get_database():

    # client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    # db = client["chatbot"]
    # client = MongoClient("mongodb://localhost:27017/", server_api=ServerApi('1'))
    # db = client["chatbot_saas"]
    mongi_uri = "mongodb://admin:M0nG0%40dm1n%212025%24@16.16.57.62:27017/admin?authSource=admin"

    client = MongoClient(mongi_uri, server_api=ServerApi('1'))
    db = client["My-portfolio"]

    return db


db = get_database()
