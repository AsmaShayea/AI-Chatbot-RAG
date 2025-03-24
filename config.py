import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_jwt_secret")
    MONGO_URI = "mongodb://localhost:27017/chatbot_saas"
