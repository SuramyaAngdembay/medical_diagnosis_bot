class DifferentialFormatter:
    def __init__(self):
        self.condition_mapping = self.load_condition_mapping()
        
    def format_differential(self, patho_pred):
        """Format BASD pathology predictions into differential diagnosis"""
        top_k = 5
        top_probs, top_indices = torch.topk(patho_pred, top_k)
        
        return [
            {
                'condition': self.condition_mapping[idx],
                'probability': float(prob),
                'confidence': self.calculate_confidence(prob)
            }
            for prob, idx in zip(top_probs, top_indices)
        ]