<<<<<<< HEAD
# medical_diagnosis_bot
=======
# Medical Diagnosis System

An AI-driven medical support system that helps patients record symptoms, summarizes medical information, and compares them against standard-of-care guidelines.

## Project Overview

This system aims to:
- Help patients record their symptoms and medical history
- Generate concise summaries of patient information
- Compare symptoms against standard medical guidelines
- Flag missing or incomplete information
- Support multi-language communication between patients and healthcare providers

## Project Structure

```
medical_diagnosis_system/
├── data/
│   ├── release_evidences.json
│   ├── release_conditions.json
│   ├── release_train_patients.csv
│   └── release_test_patients.csv
├── src/
│   ├── data_processor.py
│   ├── diagnosis_model.py
│   ├── text_processor.py
│   ├── question_selector.py
│   ├── chatbot.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python src/main.py
   ```

## Features

- Symptom recording and analysis
- Medical history tracking
- AI-powered diagnosis suggestions
- Standard-of-care guideline comparison
- Multi-language support
- Risk assessment and flagging

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
>>>>>>> 7f9267b (Initial commit: Project structure and basic setup)
