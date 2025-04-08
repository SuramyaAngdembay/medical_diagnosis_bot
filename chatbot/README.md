# Medical Diagnosis Chatbot

A simple medical diagnosis chatbot that uses the Gemini API to help users understand their symptoms.

## Overview

This is a basic implementation of a medical diagnosis chatbot that uses the Gemini API to generate responses to user inputs. The chatbot can help users understand their symptoms and provide possible diagnoses.

## Features

- Simple conversation interface
- Integration with Gemini API
- Basic medical diagnosis capabilities

## Requirements

- Python 3.8+
- Google Gemini API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical_diagnosis_bot.git
   cd medical_diagnosis_bot
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

## Usage

### Command Line Interface

Run the chatbot in CLI mode:
```bash
python -m chatbot.cli
```

### As a Module

```python
from chatbot.chatbot import MedicalChatbot

# Initialize the chatbot
chatbot = MedicalChatbot()

# Start a conversation
response = chatbot.start_conversation()
print(response)

# Process user input
user_input = "I have a headache and fever"
response = chatbot.process_user_input(user_input)
print(response)
```

## Extending the Chatbot

The chatbot is designed to be easily extended. Here are some ways you can enhance it:

1. Add a clarifier module to handle vague user inputs
2. Integrate with a reinforcement learning model for better question selection
3. Add a knowledge base for medical conditions and symptoms
4. Implement a more sophisticated conversation flow
5. Add a web interface

## Disclaimer

This chatbot is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. 