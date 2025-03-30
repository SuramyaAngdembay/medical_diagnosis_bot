# Medical Diagnosis Chatbot

An AI-driven medical diagnosis system that helps determine potential diagnoses based on symptoms and medical history using a combination of Reinforcement Learning (RL), Bayesian Automatic Symptoms Diagnosis (BASD), and Large Language Models (LLMs).

## Project Overview

This system aims to:
- Guide users through a structured medical conversation
- Collect symptoms and medical history intelligently using RL-based question selection
- Process natural language responses using Google's Gemini API
- Generate diagnoses with confidence levels using trained models
- Provide early disease prediction based on partial symptom information
- Summarize patient interactions for healthcare professionals

## Key Components and Approaches

### Reinforcement Learning for Question Selection

The RL component of our system is designed to optimize the diagnostic process by intelligently selecting the most informative questions to ask patients:

- **Policy Gradient Approach**: Our RL agent employs a policy gradient method to learn which questions will maximize information gain at each interaction step.
- **State Representation**: The patient state is represented as a vector encoding demographic information and symptom evidence.
- **Dual Network Architecture**: 
  - The policy network selects the next optimal question from 223 possible medical evidence inquiries
  - The classifier network predicts likely diagnoses based on current evidence
- **Reward Mechanism**: The agent is rewarded for reaching correct diagnoses with minimal questions, penalized for unnecessary questions
- **Exploration-Exploitation Balance**: The model balances exploring new symptom paths versus exploiting known diagnostic patterns

The RL agent dramatically improves the efficiency of symptom collection, reducing the number of questions needed while maintaining or improving diagnostic accuracy.

### Bayesian Automatic Symptoms Diagnosis (BASD)

The BASD component leverages probabilistic reasoning to infer disease likelihood:

- **Bayesian Network**: Models diseases and symptoms as a probabilistic graph with conditional dependencies
- **Prior Knowledge Integration**: Incorporates medical domain knowledge about symptom-disease relationships
- **Uncertainty Handling**: Explicitly represents diagnostic uncertainty with probability distributions
- **Incremental Updating**: Updates disease probabilities with each new piece of evidence
- **Confidence Estimation**: Provides confidence scores for diagnoses based on available evidence

BASD is particularly valuable for early-stage diagnosis when only partial information is available, helping identify concerning conditions that warrant further investigation.

### Chatbot Interface and Natural Language Processing

The chatbot serves as the user-facing component, providing several key capabilities:

- **Natural Language Understanding**: Uses Gemini API to interpret free-text patient responses
- **Medical Concept Mapping**: Maps patient descriptions to structured medical evidence codes
- **Adaptive Questioning**: Dynamically adjusts questions based on previous responses
- **Summary Generation**: Creates concise summaries of patient interactions for healthcare providers
- **Early Warning System**: Flags potentially serious conditions even with limited information
- **Explainable Recommendations**: Provides reasoning for diagnostic suggestions

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
├── basd/                      # Bayesian Automatic Symptoms Diagnosis
│   ├── asd.py                 # BASD model implementation
│   └── asd_model.py           # Neural network components for BASD
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

1. **Diagnostic Conversation**: The chatbot initiates a conversation, collecting basic demographic information.

2. **Intelligent Question Selection**: The RL agent analyzes the current state and selects the most informative next question based on thousands of training simulations.

3. **Natural Language Processing**: Patient responses are processed by Gemini API and mapped to structured evidence data.

4. **Dual Diagnostic Approaches**:
   - The RL classifier network provides diagnosis predictions based on symptom patterns
   - The BASD component uses Bayesian inference to calculate disease probabilities

5. **Early Disease Prediction**: Even with limited information, the system can identify potential conditions that match the current symptom profile, prioritizing serious conditions that require urgent attention.

6. **Summarization**: The chatbot generates a structured summary of the interaction, highlighting key findings, potential diagnoses, and confidence levels.

7. **Handoff to Healthcare Professionals**: The system is designed to augment, not replace, medical expertise. It provides its analysis to healthcare professionals for definitive diagnosis.

## Clinical Benefits

- **Reduced Time to Diagnosis**: By asking targeted questions, the system helps reach preliminary diagnoses more efficiently.
- **Consistent Assessment**: Provides standardized questioning regardless of provider fatigue or bias.
- **Increased Access**: Makes preliminary medical assessment available in underserved areas.
- **Early Warning System**: Can flag potentially serious conditions early in the assessment process.
- **Documentation**: Creates structured records of patient-reported symptoms and history.

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

### Reinforcement Learning Model Training

The RL model is trained through simulated patient interactions:
- **Patient Simulator**: Generates synthetic patient cases based on real medical data
- **Episode-Based Training**: The agent learns through thousands of complete diagnostic episodes
- **State Abstraction**: Critical medical states are encoded as vectors for efficient processing
- **Transfer Learning**: Pre-trained on large medical datasets before fine-tuning
- **Hyperparameter Optimization**: Learning rate, discount factor, and exploration parameters optimized for medical diagnosis

### NLP Integration

The system uses Google's Gemini 2.5 Pro API to:
- Process unstructured patient responses
- Map responses to structured evidence codes
- Handle ambiguous or complex responses
- Generate natural language explanations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
