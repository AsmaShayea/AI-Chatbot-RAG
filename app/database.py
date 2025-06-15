from config import MONGO_URI
from pymongo import MongoClient
from flask_pymongo import PyMongo
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
import os

def get_database():

    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = client["My-portfolio"]

    return db


db = get_database()
