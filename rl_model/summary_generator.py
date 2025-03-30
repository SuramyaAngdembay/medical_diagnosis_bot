class DoctorSummaryGenerator:
    def __init__(self):
        self.medical_knowledge_base = self.load_medical_knowledge()
        
    def generate_summary(self, differential_diagnosis, conversation_history):
        """Generate comprehensive doctor summary"""
        return {
            'primary_diagnosis': self.format_primary_diagnosis(differential_diagnosis[0]),
            'differential_diagnosis': self.format_differential(differential_diagnosis[1:]),
            'key_findings': self.extract_key_findings(conversation_history),
            'recommendations': self.generate_recommendations(differential_diagnosis),
            'risk_factors': self.identify_risk_factors(conversation_history)
        }
        
    def format_primary_diagnosis(self, diagnosis):
        """Format primary diagnosis with supporting evidence"""
        return {
            'condition': diagnosis['condition'],
            'probability': diagnosis['probability'],
            'supporting_symptoms': self.get_supporting_symptoms(diagnosis['condition']),
            'severity': self.assess_severity(diagnosis['condition'])
        }