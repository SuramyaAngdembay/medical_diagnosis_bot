from .data_processor import EvidenceLoader, DataProcessor
from .question_selector import QuestionSelector
from .diagnosis_model import MockModel
from .gemini_client import GeminiClient
import logging
import numpy as np
import torch
from typing import Optional, Tuple, Dict


logger = logging.getLogger(__name__)

class ChatbotEngine:
    def __init__(self, evidence_path: str, rl_agent=None):
        """Initialize the chatbot engine.
        
        Args:
            evidence_path: Path to the evidence metadata file
            rl_agent: Optional RL agent for question selection
        """
        try:
            self.loader = EvidenceLoader(evidence_path)
            self.data_processor = DataProcessor(evidence_path)
            
            # Create a mock model if no RL agent provided
            self.model = MockModel() if rl_agent is None else rl_agent
            self.question_selector = QuestionSelector(self.loader, self.model)
            
            self.rl_agent = rl_agent
            self.answers = {}
            self.turn_count = 0
            self.max_turns = 20
            
            # Initialize demographic information
            self.demographics = {
                "age": None,
                "sex": None,
                "race": None,
                "ethnicity": None,
                "turn": self.turn_count,
                "max_turns": self.max_turns
            }
            
            self.gemini_client = GeminiClient()
            logger.info("ChatbotEngine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatbotEngine: {str(e)}")
            raise

    def start_conversation(self, initial_demographics: Dict[str, any] = None):
        """Start a new conversation.
        
        Args:
            initial_demographics: Optional dictionary with initial demographic information
        """
        self.answers = {}
        self.turn_count = 0
        
        # Reset demographics with provided values or defaults
        self.demographics = {
            "age": 30,
            "sex": "M",
            "race": "other",
            "ethnicity": "nonhispanic",
            "turn": self.turn_count,
            "max_turns": self.max_turns
        }
        if initial_demographics:
            self.demographics.update(initial_demographics)
            
    def start_session(self, initial_demographics: Dict[str, any] = None):
        """Alias for start_conversation to maintain API compatibility."""
        return self.start_conversation(initial_demographics)
            
    def update_demographics(self, demographics: Dict[str, any]):
        """Update demographic information.
        
        Args:
            demographics: Dictionary containing demographic updates
        """
        self.demographics.update(demographics)
        
    def get_current_state(self) -> np.ndarray:
        """Get the current state vector including demographics."""
        # Update turn count in demographics
        self.demographics["turn"] = self.turn_count
        return self.data_processor.get_state_from_answers(self.answers, self.demographics)
        
    def process_answer(self, evidence_code: str, value: any) -> bool:
        """Process an answer from the user.
        
        Args:
            evidence_code: The evidence code being answered
            value: The answer value
            
        Returns:
            bool: True if the answer was valid and processed
        """
        if self.data_processor.validate_evidence_structure(evidence_code, value):
            self.answers[evidence_code] = value
            # Record answer in question selector
            self.question_selector.record_answer(evidence_code, value)
            return True
        return False
        
    def ask_next(self) -> Tuple[Optional[str], Optional[Dict]]:
        """Get the next question to ask using the RL agent or fallback selector.
        
        Returns:
            Tuple[Optional[str], Optional[Dict]]: The next evidence code and its data, or (None, None) if done
        """
        try:
            if self.turn_count >= self.max_turns:
                logger.info("Maximum turns reached")
                return None, {"error": "Maximum turns reached"}
                
            self.turn_count += 1
            self.demographics["turn"] = self.turn_count
            
            if self.rl_agent:
                try:
                    # Get current state with demographics
                    current_state = self.get_current_state()
                    
                    # Use RL agent to select next question
                    action = self.rl_agent.choose_action_s(current_state.reshape(1, -1))
                    if isinstance(action, np.ndarray):
                        action = action[0]  # Get scalar from array
                        
                    # Convert action to evidence code
                    evidence_codes = self.data_processor.evidence_codes
                    if 0 <= action < len(evidence_codes):
                        next_evidence = evidence_codes[action]
                        # Check if already asked
                        if next_evidence not in self.answers:
                            logger.info(f"RL agent selected evidence code: {next_evidence}")
                            evidence_data = self.loader.get_evidence_data(next_evidence)
                            return next_evidence, evidence_data
                            
                    logger.warning("RL agent selected invalid or repeated question, falling back to selector")
                    
                except Exception as e:
                    logger.error(f"Error using RL agent: {str(e)}")
                    logger.warning("Falling back to question selector")
            
            # Fallback to basic selector
            next_evidence = self.question_selector.next_question()
            if next_evidence:
                evidence_data = self.loader.get_evidence_data(next_evidence)
                return next_evidence, evidence_data
            return None, {"error": "No more questions available"}
            
        except Exception as e:
            logger.error(f"Error in ask_next: {str(e)}")
            return None, {"error": str(e)}
            
    def get_diagnosis(self) -> Tuple[int, float]:
        """Get the current diagnosis using the RL agent.
        
        Returns:
            Tuple[int, float]: (diagnosis_index, confidence)
        """
        if self.rl_agent:
            try:
                current_state = self.get_current_state()
                diagnosis_idx, probabilities = self.rl_agent.choose_diagnosis(current_state.reshape(1, -1))
                if isinstance(diagnosis_idx, np.ndarray):
                    diagnosis_idx = diagnosis_idx[0]
                confidence = float(probabilities[0][diagnosis_idx])
                return diagnosis_idx, confidence
            except Exception as e:
                logger.error(f"Error getting diagnosis: {str(e)}")
                
        return -1, 0.0  # Default values if no diagnosis available

    def submit_answer(self, code, user_input):
        """Submit and process a user's answer."""
        try:
            if not code or not user_input:
                return {"error": "Missing code or answer"}

            if self.data_processor is None:
                return {"error": "No active session. Please start a session first."}

            evidence_data = self.loader.get_evidence_data(code)
            if not evidence_data:
                return {"error": "Invalid evidence code"}

            data_type = evidence_data.get("data_type", "B")  # Default to binary
            print(f"Data type for code {code}: {data_type}") # Debug print
            print(f"Evidence data: {evidence_data}")  # Debug print

            if data_type == "B":
                # Handle binary questions using Yes/No
                if user_input.lower() not in ["yes", "no"]:
                    # Use Gemini to interpret unclear responses
                    gemini_response = self.gemini_client.generate_response(
                        evidence_data.get("question_en"),
                        user_input,
                        ['yes', 'no']
                    )
                    if gemini_response == "NO_MATCH":
                        return {"error": "Unable to interpret response."}
                    answer = 1 if gemini_response.lower() == "yes" else -1
                else:
                    answer = 1 if user_input.lower() == "yes" else -1
                self.answers[code] = answer
                return {"message": "Answer recorded successfully"}
            # Handle Categorical Questions using Gemini
            elif data_type == "C":
                # Get possible values and their meanings
                value_meaning = evidence_data.get("value_meaning", {})
                possible_values = evidence_data.get("possible-values", [])
                print(f"Value meaning: {value_meaning}")  # Debug print
                print(f"Possible values: {possible_values}")  # Debug print
                
                if not possible_values:
                    return {"error": "No valid options found for this question."}

                # Determine if this is a numeric scale or predefined categories
                is_numeric_scale = all(str(x).isdigit() for x in possible_values)
                
                # Handle empty value_meaning differently based on type
                if not value_meaning:
                    if is_numeric_scale:
                        # For numeric scales, create descriptive meanings
                        min_val = min(possible_values)
                        max_val = max(possible_values)
                        value_meaning = {
                            str(i): {
                                "en": f"Scale {i} ({i} = {'lowest' if i == min_val else 'highest' if i == max_val else 'moderate'})"
                            } for i in possible_values
                        }
                    else:
                        # For predefined categories without meanings, use values as is
                        value_meaning = {str(val): {"en": str(val)} for val in possible_values}

                # Create options list for Gemini
                options = []
                for key, value in value_meaning.items():
                    if isinstance(value, dict) and "en" in value:
                        # Skip NA values as they shouldn't be valid options for the user
                        if value["en"].upper() != "NA":
                            if is_numeric_scale:
                                # For numeric scales, just use the number
                                options.append(str(key))
                            else:
                                # For categories, use the English meaning
                                options.append(value["en"])
                    elif isinstance(value, str) and value.upper() != "NA":
                        options.append(value)
                
                print(f"Options for Gemini: {options}")  # Debug print

                if not options:
                    return {"error": "No valid options found for this question."}

                # Get the question text
                question = evidence_data.get("question_en", "")
                if not question:
                    return {"error": "Question text not found."}

                # Add context for numeric scales
                if is_numeric_scale:
                    min_val = min(possible_values)
                    max_val = max(possible_values)
                    question = f"{question} (On a scale from {min_val} to {max_val}, where {min_val} is lowest and {max_val} is highest)"

                # Use Gemini to interpret the response
                gemini_response = self.gemini_client.generate_response(
                    question,
                    user_input,
                    options
                )
                print(f"Gemini response: {gemini_response}")  # Debug print

                if gemini_response == "NO_MATCH":
                    return {"error": "Unable to determine a valid response using AI."}

                # Find the value ID that matches the Gemini response
                try:
                    answer_key = None
                    if is_numeric_scale:
                        # For numeric scales, use the response directly if it's a valid number
                        if gemini_response in [str(x) for x in possible_values]:
                            answer_key = gemini_response
                    else:
                        # For categories, match the English meaning to the key
                        for key, value in value_meaning.items():
                            if isinstance(value, dict) and "en" in value:
                                if value["en"].lower() == gemini_response.lower():
                                    answer_key = key
                                    break
                    
                    if answer_key is None:
                        return {"error": "Invalid categorical answer"}
                        
                    self.answers[code] = answer_key
                    logger.info(f"Categorical answer recorded for code {code}: {answer_key}")
                    return {"message": "Answer recorded successfully"}
                except Exception as e:
                    logger.error(f"Error processing Gemini response: {str(e)}")
                    return {"error": "Failed to process the response"}

            else:
                return {"error": f"Unsupported data type: {data_type}"}

        except Exception as e:
            logger.error(f"Error submitting answer: {str(e)}")
            return {"error": str(e)}


    def get_evidence_vector(self):
        """Get the current evidence vector."""
        try:
            if self.data_processor is None:
                logger.warning("No active session. Starting new session...")
                self.start_conversation()

            if self.data_processor is None:
                logger.error("Failed to initialize data processor")
                return {}

            return self.data_processor.get_evidence_vector()
        except Exception as e:
            logger.error(f"Error getting evidence vector: {str(e)}")
            return {}