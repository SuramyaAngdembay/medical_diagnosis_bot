from data_processor import EvidenceLoader
from question_selector import QuestionSelector
from diagnosis_model import MockModel
import logging

logger = logging.getLogger(__name__)

class ChatbotEngine:
    def __init__(self, evidence_path):
        
        try:
            self.loader = EvidenceLoader(evidence_path)
            self.model = None
            self.selector = None
            self.answers = {}
            logger.info("ChatbotEngine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatbotEngine: {str(e)}")
            raise

    def start_session(self):
        """Start a new chat session."""
        try:
            if self.model is None:
                self.model = MockModel()
                logger.info("Model initialized")
            if self.selector is None:
                self.selector = QuestionSelector(self.loader, self.model)
                logger.info("Question selector initialized")
            self.answers = {}
            logger.info("New session started")
        except Exception as e:
            logger.error(f"Failed to start session: {str(e)}")
            raise

    def ask_next(self):
        """Get the next question to ask the user."""
        try:
            if self.selector is None:
                logger.warning("No active session. Starting new session...")
                self.start_session()
                
            if self.selector is None:
                logger.error("Failed to initialize question selector")
                return None, {"error": "Failed to initialize question selector"}
                
            code = self.selector.next_question()
            if code is None:
                logger.info("No more questions to ask")
                return None, None
                
            try:
                question = self.loader.get_question(code)
                values = self.loader.get_possible_values(code)
                return code, {"question": question, "values": values}
            except Exception as e:
                logger.error(f"Error getting question data for code {code}: {str(e)}")
                return None, {"error": f"Failed to get question data: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Error in ask_next: {str(e)}")
            return None, {"error": str(e)}

    def submit_answer(self, code, user_input):
        """Submit and process a user's answer."""
        try:
            if not code or not user_input:
                return {"error": "Missing code or answer"}
                
            if self.selector is None:
                return {"error": "No active session. Please start a session first."}
                
            evidence_data = self.loader.get_evidence_data(code)
            if not evidence_data:
                return {"error": "Invalid evidence code"}

            data_type = evidence_data.get("data_type", "B")  # Default to binary if not specified

            if data_type == "B":
                if user_input.lower() not in ["yes", "no"]:
                    return {"error": "Invalid answer for binary question. Must be 'yes' or 'no'"}
                answer = 1 if user_input.lower() == "yes" else -1
                self.answers[code] = answer
                logger.info(f"Answer recorded for code {code}: {answer}")
                return {"message": "Answer recorded successfully"}
            else:
                return {"error": f"Unsupported data type: {data_type}"}
                
        except Exception as e:
            logger.error(f"Error submitting answer: {str(e)}")
            return {"error": str(e)}

    def get_evidence_vector(self):
        """Get the current evidence vector."""
        try:
            if self.selector is None:
                logger.warning("No active session. Starting new session...")
                self.start_session()
                
            if self.selector is None:
                logger.error("Failed to initialize question selector")
                return {}
                
            return self.selector.get_evidence_vector()
        except Exception as e:
            logger.error(f"Error getting evidence vector: {str(e)}")
            return {}
