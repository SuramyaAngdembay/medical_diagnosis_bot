"""
Medical Diagnosis Chatbot

A simple chatbot for medical diagnosis using the Gemini API.
"""

import os
import sys
from typing import Dict, List, Optional

# Add the parent directory to the path to import the llm_handler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_handler import GeminiAPI


class MedicalChatbot:
    """
    A simple medical diagnosis chatbot.
    
    This chatbot uses the Gemini API to generate responses to user inputs.
    """
    
    def __init__(self):
        """
        Initialize the chatbot.
        """
        self.llm = GeminiAPI()
        self.conversation_history = []
    
    def start_conversation(self) -> str:
        """
        Start a new conversation with the user.
        
        Returns:
            The initial greeting message.
        """
        self.conversation_history = []
        return "Hello! I'm your medical diagnosis assistant. How can I help you today?"
    
    def process_user_input(self, user_input: str) -> str:
        """
        Process user input and return a response.
        
        Args:
            user_input: The user's input text.
            
        Returns:
            The chatbot's response.
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate a response using the LLM
        response = self._generate_response(user_input)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _generate_response(self, user_input: str) -> str:
        """
        Generate a response to the user's input.
        
        Args:
            user_input: The user's input text.
            
        Returns:
            The generated response.
        """
        # Create a prompt for the LLM
        prompt = f"""
        You are a medical diagnosis assistant. The user has provided the following input:
        
        {user_input}
        
        Please provide a helpful response. If the user is describing symptoms, ask relevant follow-up questions.
        If you have enough information, provide a possible diagnosis and recommendations.
        
        Respond in a friendly, empathetic tone.
        """
        
        try:
            # Generate a response using the LLM
            response = self.llm.generate_content(prompt)
            return self.llm.extract_text(response)
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error. Please try again." 