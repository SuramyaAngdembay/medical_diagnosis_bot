class LLMToBASDConverter:
    def __init__(self):
        self.symptom_mapping = self.load_symptom_mapping()
        self.value_mapping = self.load_value_mapping()
        
    def convert_to_basd_format(self, llm_output):
        """Convert LLM output to BASD input format"""
        return {
            'x_sym': self.convert_symptoms(llm_output['symptoms']),
            'x_ag': self.convert_actions(llm_output['actions']),
            'patient_info': self.convert_patient_info(llm_output['patient_info'])
        }
        
    def convert_symptoms(self, symptoms):
        """Convert symptoms to BASD format"""
        return torch.tensor([
            self.symptom_mapping.get(symptom, 0)
            for symptom in symptoms
        ])