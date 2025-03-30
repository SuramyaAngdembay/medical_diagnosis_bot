# from data_processor import EvidenceLoader
# from question_selector import QuestionSelector
# from diagnosis_model import MockModel
# from gemini_client import GeminiClient
# import logging


# logger = logging.getLogger(__name__)

# class ChatbotEngine:
#     def __init__(self, evidence_path):
        
#         try:
#             self.loader = EvidenceLoader(evidence_path)
#             self.model = None
#             self.selector = None
#             self.answers = {}
#                 self.gemini_client = GeminiClient()
#             logger.info("ChatbotEngine initialized successfully")
#         except Exception as e:
#             logger.error(f"Failed to initialize ChatbotEngine: {str(e)}")
#             raise

#     def start_session(self):
#         """Start a new chat session."""
#         try:
#             if self.model is None:
#                 self.model = MockModel()
#                 logger.info("Model initialized")
#             if self.selector is None:
#                 self.selector = QuestionSelector(self.loader, self.model)
#                 logger.info("Question selector initialized")
#             self.answers = {}
#             logger.info("New session started")
#         except Exception as e:
#             logger.error(f"Failed to start session: {str(e)}")
#             raise

#     def ask_next(self):
#         """Get the next question to ask the user."""
#         try:
#             if self.selector is None:
#                 logger.warning("No active session. Starting new session...")
#                 self.start_session()
                
#             if self.selector is None:
#                 logger.error("Failed to initialize question selector")
#                 return None, {"error": "Failed to initialize question selector"}
                
#             code = self.selector.next_question()
#             if code is None:
#                 logger.info("No more questions to ask")
#                 return None, None
                
#             try:
#                 question = self.loader.get_question(code)
#                 values = self.loader.get_possible_values(code)
#                 return code, {"question": question, "values": values}
#             except Exception as e:
#                 logger.error(f"Error getting question data for code {code}: {str(e)}")
#                 return None, {"error": f"Failed to get question data: {str(e)}"}
                
#         except Exception as e:
#             logger.error(f"Error in ask_next: {str(e)}")
#             return None, {"error": str(e)}

#     # def submit_answer(self, code, user_input):
#     #     """Submit and process a user's answer."""
#     #     try:
#     #         if not code or not user_input:
#     #             return {"error": "Missing code or answer"}
                
#     #         if self.selector is None:
#     #             return {"error": "No active session. Please start a session first."}
                
#     #         evidence_data = self.loader.get_evidence_data(code)
#     #         if not evidence_data:
#     #             return {"error": "Invalid evidence code"}

#     #         data_type = evidence_data.get("data_type", "B")  # Default to binary if not specified

#     #         if data_type == "B":
#     #             if user_input.lower() not in ["yes", "no"]:
#     #                 return {"error": "Invalid answer for binary question. Must be 'yes' or 'no'"}
#     #             answer = 1 if user_input.lower() == "yes" else -1
#     #             self.answers[code] = answer
#     #             logger.info(f"Answer recorded for code {code}: {answer}")
#     #             return {"message": "Answer recorded successfully"}
#     #         else:
#     #             return {"error": f"Unsupported data type: {data_type}"}
                
#     #     except Exception as e:
#     #         logger.error(f"Error submitting answer: {str(e)}")
#     #         return {"error": str(e)}

#     def submit_answer(self, code, user_input):
#         """Submit and process a user's answer."""
#         try:
#             if not code or not user_input:
#                 return {"error": "Missing code or answer"}

#             if self.selector is None:
#                 return {"error": "No active session. Please start a session first."}

#             evidence_data = self.loader.get_evidence_data(code)
#             if not evidence_data:
#                 return {"error": "Invalid evidence code"}

#             data_type = evidence_data.get("data_type", "B")  # Default to binary
#             print(f"Data type for code {code}: {data_type}") # Debug print

#             if data_type == "B":
#                 # Handle binary questions using Yes/No
#                 if user_input.lower() not in ["yes", "no"]:
#                     # Use Gemini to interpret unclear responses
#                     gemini_response = self.gemini_client.generate_response(
#                         evidence_data.get("question_en"),
#                         user_input,
#                         ['yes', 'no']
#                     )
#                     if gemini_response == "NO_MATCH":
#                         return {"error": "Unable to interpret response."}
#                     answer = 1 if gemini_response.lower() == "yes" else -1
#                 else:
#                     answer = 1 if user_input.lower() == "yes" else -1
#                 self.answers[code] = answer
#                 return {"message": "Answer recorded successfully"}
#             # Handle Categorical Questions using Gemini
#             elif data_type == "C":
#                 # Get possible values and their meanings
#                 value_meaning = evidence_data.get("value_meaning", {})
#                 possible_values = evidence_data.get("possible-values", [])
#                 print(f"Value meaning: {value_meaning}")  # Debug print
#                 print(f"Possible values: {possible_values}")  # Debug print
                
#                 if not possible_values:
#                     return {"error": "No valid options found for this question."}

#                 # Determine if this is a numeric scale or predefined categories
#                 is_numeric_scale = all(str(x).isdigit() for x in possible_values)
                
#                 # Handle empty value_meaning differently based on type
#                 if not value_meaning:
#                     if is_numeric_scale:
#                         # For numeric scales, create descriptive meanings
#                         min_val = min(possible_values)
#                         max_val = max(possible_values)
#                         value_meaning = {
#                             str(i): {
#                                 "en": f"Scale {i} ({i} = {'lowest' if i == min_val else 'highest' if i == max_val else 'moderate'})"
#                             } for i in possible_values
#                         }
#                     else:
#                         # For predefined categories without meanings, use values as is
#                         value_meaning = {str(val): {"en": str(val)} for val in possible_values}

#                 # Create options list for Gemini
#                 options = []
#                 for key, value in value_meaning.items():
#                     if isinstance(value, dict) and "en" in value:
#                         # Skip NA values as they shouldn't be valid options for the user
#                         if value["en"].upper() != "NA":
#                             if is_numeric_scale:
#                                 # For numeric scales, just use the number
#                                 options.append(str(key))
#                             else:
#                                 # For categories, use the English meaning
#                                 options.append(value["en"])
#                     elif isinstance(value, str) and value.upper() != "NA":
#                         options.append(value)
                
#                 print(f"Options for Gemini: {options}")  # Debug print

#                 if not options:
#                     return {"error": "No valid options found for this question."}

#                 # Get the question text
#                 question = evidence_data.get("question_en", "")
#                 if not question:
#                     return {"error": "Question text not found."}

#                 # Add context for numeric scales
#                 if is_numeric_scale:
#                     min_val = min(possible_values)
#                     max_val = max(possible_values)
#                     question = f"{question} (On a scale from {min_val} to {max_val}, where {min_val} is lowest and {max_val} is highest)"

#                 # Use Gemini to interpret the response
#                 gemini_response = self.gemini_client.generate_response(
#                     question,
#                     user_input,
#                     options
#                 )
#                 print(f"Gemini response: {gemini_response}")  # Debug print

#                 if gemini_response == "NO_MATCH":
#                     return {"error": "Unable to determine a valid response using AI."}

#                 # Find the value ID that matches the Gemini response
#                 try:
#                     answer_key = None
#                     if is_numeric_scale:
#                         # For numeric scales, use the response directly if it's a valid number
#                         if gemini_response in [str(x) for x in possible_values]:
#                             answer_key = gemini_response
#                     else:
#                         # For categories, match the English meaning to the key
#                         for key, value in value_meaning.items():
#                             if isinstance(value, dict) and "en" in value:
#                                 if value["en"].lower() == gemini_response.lower():
#                                     answer_key = key
#                                     break
#                     
#                     if answer_key is None:
#                         return {"error": "Invalid categorical answer"}
#                         
#                     self.answers[code] = answer_key
#                     logger.info(f"Categorical answer recorded for code {code}: {answer_key}")
#                     return {"message": "Answer recorded successfully"}
#                 except Exception as e:
#                     logger.error(f"Error processing Gemini response: {str(e)}")
#                     return {"error": "Failed to process the response"}

#             else:
#                 return {"error": f"Unsupported data type: {data_type}"}

#         except Exception as e:
#             logger.error(f"Error submitting answer: {str(e)}")
#             return {"error": str(e)}


#     def get_evidence_vector(self):
#         """Get the current evidence vector."""
#         try:
#             if self.selector is None:
#                 logger.warning("No active session. Starting new session...")
#                 self.start_session()
                
#             if self.selector is None:
#                 logger.error("Failed to initialize question selector")
#                 return {}
                
#             return self.selector.get_evidence_vector()
#         except Exception as e:
#             logger.error(f"Error getting evidence vector: {str(e)}")
#             return {}



from data_processor import EvidenceLoader
from question_selector import QuestionSelector
from diagnosis_model import MockModel
from gemini_client import GeminiClient
import logging


logger = logging.getLogger(__name__)

class ChatbotEngine:
    def __init__(self, evidence_path):

        try:
            self.loader = EvidenceLoader(evidence_path)
            self.model = None
            self.selector = None
            self.answers = {}
            self.gemini_client = GeminiClient()
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