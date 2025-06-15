from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)

    # ✅ Enable CORS for your frontend domain
    CORS(app, origins=["https://asmashayea.com"])

    # Register your API blueprint
    from app.routes import api_bp
    app.register_blueprint(api_bp)

    return app
