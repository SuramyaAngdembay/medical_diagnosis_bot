from flask import Flask, request, jsonify
from chatbot import ChatbotEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    chatbot = ChatbotEngine('data/release_evidences.json')
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    raise

@app.route("/")
def home():
    """Welcome endpoint."""
    return jsonify({
        "message": "Welcome to the Medical Diagnosis Chatbot!",
        "status": "running"
    })

@app.route("/start", methods=["POST"])
def start_session():
    """Start a new chat session."""
    try:
        chatbot.start_session()
        return jsonify({"message": "Session started successfully"})
    except Exception as e:
        logger.error(f"Failed to start session: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/question", methods=["GET"])
def get_question():
    """Get the next question to ask the user."""
    try:
        if chatbot.selector is None:
            chatbot.start_session()

        code, question_data = chatbot.ask_next()
        if code is None:
            if question_data and "error" in question_data:
                return jsonify(question_data), 400
            return jsonify({
                "message": "Diagnosis complete or confidence reached",
                "status": "complete"
            })
        return jsonify({"code": code, **question_data})
    except Exception as e:
        logger.error(f"Error getting question: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/answer", methods=["POST"])
def submit_answer():
    """Submit a user's answer to a question."""
    try:
        if chatbot.selector is None:
            return jsonify({
                "error": "Session not started. Please call /start first."
            }), 400

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        code = data.get("code")
        user_input = data.get("answer")
        
        if not code or not user_input:
            return jsonify({
                "error": "Missing code or answer in request"
            }), 400

        result = chatbot.submit_answer(code, user_input)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error submitting answer: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/confidence", methods=["GET"])
def get_confidence():
    """Get the current diagnosis confidence levels."""
    # Get current confidence level once model is implementeds!

if __name__ == "__main__":
    app.run(debug=True)
