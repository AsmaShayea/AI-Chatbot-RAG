from flask import Flask
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from app.database import db

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "your_secret_key"

    CORS(app)
    JWTManager(app)

    from app.routes import api_bp
    app.register_blueprint(api_bp)

    return app
