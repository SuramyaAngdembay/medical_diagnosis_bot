class LLMHandler:
    def __init__(self):
        self.llm = LLMClient()  # Initialize your LLM (ChatGPT/Gemini)
        
    def process_user_input(self, user_input):
        """Process raw user input through LLM"""
        return {
            'symptoms': self.extract_symptoms(user_input),
            'patient_info': self.extract_patient_info(user_input),
            'context': self.extract_context(user_input)
        }
        
    def extract_symptoms(self, text):
        """Extract symptoms from text"""
        prompt = f"Extract medical symptoms from: {text}"
        return self.llm.analyze(prompt)
        
    def extract_patient_info(self, text):
        """Extract patient information"""
        prompt = f"Extract patient demographics and medical history from: {text}"
        return self.llm.analyze(prompt)