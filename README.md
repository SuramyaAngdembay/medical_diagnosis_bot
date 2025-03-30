# Medical Diagnosis Chatbot

An AI-driven medical diagnosis system that helps determine potential diagnoses based on symptoms and medical history using a combination of Reinforcement Learning (RL) and Large Language Models (LLMs).

## Project Overview

This system aims to:
- Guide users through a structured medical conversation
- Collect symptoms and medical history intelligently using RL-based question selection
- Process natural language responses using Google's Gemini API
- Generate diagnoses with confidence levels using trained models
- Provide a user-friendly interface for patient interaction

## Architecture

The system consists of several key components:

```
medical_diagnosis_bot/
├── data/                      # Medical evidence and condition data
│   ├── release_evidences.json # Question definitions and evidence codes
│   └── release_conditions.json # Disease/condition definitions
├── rl_model/                  # Reinforcement Learning components
│   ├── agent.py               # RL agent for question selection and diagnosis
│   └── output/                # Saved model weights
├── src/                       # Core application code
│   ├── chatbot.py             # Main chatbot engine
│   ├── data_processor.py      # Processes medical data
│   ├── Gemini_client.py       # Integration with Google's Gemini API
│   └── main.py                # Flask application entry point
├── static/                    # Web frontend assets
│   ├── css/                   # Styling
│   └── js/                    # Frontend JavaScript
├── templates/                 # HTML templates for the web interface
├── .env                       # Environment variables (API keys)
└── requirements.txt           # Project dependencies
```

## How It Works

1. **Question Selection**: The RL agent selects the most informative questions to ask based on the current state.
2. **Natural Language Processing**: User responses are processed using the Gemini API to extract structured data.
3. **State Representation**: The system maintains a state vector representing the patient's condition.
4. **Diagnosis Generation**: When sufficient information is gathered, the system provides a diagnosis with confidence levels.

## Setup and Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medical_diagnosis_bot.git
   cd medical_diagnosis_bot
   ```

2. **Set up the environment**
   ```bash
   # Create and activate a conda environment
   conda create -n medical_bot python=3.9
   conda activate medical_bot
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   Create a `.env` file in the project root with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Run the application**
   ```bash
   python -m src.main
   ```
   The web interface will be available at http://localhost:5000

## Technical Details

### Reinforcement Learning Model

The RL component uses a policy gradient approach to learn optimal question selection strategies. The model consists of:
- A policy network for selecting the next question
- A classifier network for predicting diagnoses

The agent is trained to maximize information gain while minimizing the number of questions needed for an accurate diagnosis.

### NLP Integration

The system uses Google's Gemini 2.5 Pro API to:
- Process unstructured patient responses
- Map responses to structured evidence codes
- Handle ambiguous or complex responses

## License

This project is licensed under the MIT License - see the LICENSE file for details.
