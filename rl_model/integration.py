class MedicalSystemIntegration:
    def __init__(self):
        self.llm_handler = LLMHandler()
        self.summary_generator = DoctorSummaryGenerator()
        self.aarlc_model = Policy_Gradient_pair_model(...)  # Load trained model
        self.basd_model = ASDMLP(...)  # Load trained model
        
    def process_interaction(self, user_input):
        # 1. Process through LLM
        processed_input = self.llm_handler.process_user_input(user_input)
        
        # 2. Generate differential diagnosis using BASD
        differential = self.generate_differential(processed_input)
        
        # 3. Get next question from AARLC
        next_question = self.get_next_question(differential)
        
        # 4. Generate doctor summary
        summary = self.summary_generator.generate_summary(
            differential,
            self.conversation_history
        )
        
        return {
            'next_question': next_question,
            'differential_diagnosis': differential,
            'doctor_summary': summary
        }