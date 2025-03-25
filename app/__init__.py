from flask import Flask
from flask_pymongo import PyMongo
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    # Load configuration
    app.config.from_object('config')

    # Initialize MongoDB
    mongo = PyMongo(app, uri=app.config['MONGO_URI'])

    # Register Blueprints
    from app.routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/")

    @app.route('/')
    def index():
        return "Welcome to the Chatbot App!"

    return app
