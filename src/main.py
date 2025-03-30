import logging
import os
import sys
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Change relative imports to absolute imports
from src.chatbot import ChatbotEngine
from src.data_processor import DataProcessor
from src.Gemini_client import GeminiClient  # Note the capital G in Gemini_client
from rl_model.agent import Policy_Gradient_pair_model  # Import the RL agent class

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# Specify template_folder and static_folder
app = Flask(__name__, template_folder=os.path.join(parent_dir, 'templates'), static_folder=os.path.join(parent_dir, 'static'))

# Initialize components
evidence_path = os.path.join(parent_dir, "data/release_evidences.json")
try:
    data_processor = DataProcessor(evidence_path)
    logger.info("Data processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

gemini_client = GeminiClient()

# RL Agent Loading and Chatbot Initialization
rl_agent = None
chatbot = None  # Initialize chatbot as None
try:
    # Get dimensions from data processor
    state_size = data_processor.get_state_size()
    disease_size = data_processor.get_disease_count()
    symptom_size = data_processor.get_symptom_count()

    logger.info(f"Initializing RL agent with dimensions: state={state_size}, disease={disease_size}, symptom={symptom_size}")

    # Model hyperparameters (from training)
    learning_rate = 1e-4
    gamma = 0.99
    eta = 0.01

    # Initialize model
    rl_agent = Policy_Gradient_pair_model(state_size, disease_size, symptom_size, LR=learning_rate, Gamma=gamma, Eta=eta)

    # Load pre-trained weights if available
    model_dir = os.path.join('rl_model', 'output')
    policy_checkpoint_path = os.path.join(model_dir, 'best_policy_casande_1_1.0_2.826_1.pth')
    classifier_checkpoint_path = os.path.join(model_dir, 'best_classifier_casande_1_1.0_2.826_1.pth')

    if os.path.exists(policy_checkpoint_path) and os.path.exists(classifier_checkpoint_path):
        # Load weights with correct device mapping
        device = rl_agent.device  # Use the device from RL agent
        try:
            rl_agent.policy.load_state_dict(torch.load(policy_checkpoint_path, map_location=device))
            rl_agent.classifier.load_state_dict(torch.load(classifier_checkpoint_path, map_location=device))
            logger.info(f"Pre-trained RL Agent loaded successfully on {device}")
        except Exception as e:
            logger.warning(f"Error loading model weights: {e}. Starting with untrained model.")
    else:
        logger.warning(f"Pre-trained RL Agent weights not found in {model_dir}. Starting with untrained model.")

    # Initialize chatbot with RL agent *only if rl_agent loaded successfully*
    chatbot = ChatbotEngine(evidence_path, rl_agent=rl_agent)
    logger.info("Chatbot initialized with RL agent")

except Exception as e:
    logger.error(f"Failed to load RL Agent: {e}")
    rl_agent = None  # Ensure rl_agent is None if loading fails
    # Initialize chatbot without RL agent as fallback
    chatbot = ChatbotEngine(evidence_path)
    logger.warning("Chatbot initialized without RL agent - will use fallback question selection")

@app.route("/")
def home():
    """Serve the main chatbot HTML page."""
    return render_template('index.html')

# Add route to serve static files (CSS, JS)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

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
        # Start session if not already started
        if not hasattr(chatbot, 'answers'):
            chatbot.start_session()

        code, question_data = chatbot.ask_next()
        if code is None:
            if question_data and "error" in question_data:
                logger.warning(f"Error getting question: {question_data['error']}")
                return jsonify(question_data), 400
            return jsonify({
                "message": "Diagnosis complete or confidence reached",
                "status": "complete"
            })
            
        logger.info(f"Selected question {code}: {question_data.get('question', '')}")
        return jsonify({"code": code, **question_data})
        
    except Exception as e:
        logger.error(f"Error getting question: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/answer", methods=["POST"])
def submit_answer():
    """Submit a user's answer to a question."""
    try:
        if not hasattr(chatbot, 'answers'):
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
    try:
        if not hasattr(chatbot, 'answers'):
            return jsonify({
                "error": "Session not started. Please call /start first."
            }), 400

        if rl_agent is None:
            return jsonify({
                "error": "RL model not loaded properly"
            }), 500

        # Get current state from chatbot
        current_state = chatbot.get_rl_compatible_state()
        
        # Convert to tensor and move to correct device
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(rl_agent.device)

        # Get diagnosis probabilities
        with torch.no_grad():
            diagnosis_logits = rl_agent.classifier(state_tensor)
            diagnosis_probs = torch.softmax(diagnosis_logits, dim=1)

        # Convert to list and get top diagnoses
        diagnosis_probs = diagnosis_probs[0].cpu().numpy()
        top_indices = diagnosis_probs.argsort()[-5:][::-1]  # Get top 5 diagnoses
        
        # Get disease names and create response
        diseases = data_processor.get_disease_names()
        confidence_levels = [{
            "disease": diseases[idx],
            "confidence": float(diagnosis_probs[idx])
        } for idx in top_indices]

        return jsonify({
            "confidence_levels": confidence_levels,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error getting confidence levels: {str(e)}")
        return jsonify({"error": str(e)}), 500

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Start Flask server
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)

if __name__ == "__main__":
    main()