"""
Command Line Interface

A simple CLI for testing the medical diagnosis chatbot.
"""

import os
import sys
from typing import Optional

# Add the parent directory to the path to import the components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.chatbot import MedicalChatbot


def main():
    """Run the chatbot in CLI mode."""
    print("Initializing the medical diagnosis chatbot...")
    
    # Initialize the chatbot
    chatbot = MedicalChatbot()
    
    # Start the conversation
    response = chatbot.start_conversation()
    print("\nChatbot:", response)
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nChatbot: Goodbye! Take care!")
                break
            
            # Process the input and get the response
            response = chatbot.process_user_input(user_input)
            print("\nChatbot:", response)
            
        except KeyboardInterrupt:
            print("\n\nChatbot: Goodbye! Take care!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'exit' to quit.")


if __name__ == "__main__":
    main() 