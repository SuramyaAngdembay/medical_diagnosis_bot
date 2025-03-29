import numpy as np
from scipy.stats import entropy

class QuestionSelector:
    def __init__(self, evidence_loader, model, max_questions=15, confidence_threshold=0.9):
        self.loader = evidence_loader
        self.model = model
        self.remaining = set(self.loader.get_all_evidence_codes())
        self.asked = set()
        self.answers = {}
        self.max_questions = max_questions
        self.confidence_threshold = confidence_threshold

    def next_question(self):
        # Stop if max questions are asked or confidence is high
        if len(self.asked) >= self.max_questions or self._confidence_met():
            return None
        
        best_question = self._select_best_question()
        if best_question is None:
            return None
        
        self.asked.add(best_question)
        return best_question
    
    def _confidence_met(self):
        probs = self.model.predict(self.answers)
        max_confidence = max(probs.values())
        return max_confidence >= self.confidence_threshold

    def _select_best_question(self):
        max_info_gain = -float('inf')
        best_question = None
        
        for code in self.remaining - self.asked:
            info_gain = self._calculate_information_gain(code)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_question = code
                
        return best_question
    
    def _calculate_information_gain(self, code):
        current_entropy = entropy(list(self.model.predict(self.answers).values()))

        # Simulate both possible outcomes
        answers_yes = self.answers.copy()
        answers_no = self.answers.copy()
        answers_yes[code] = 1
        answers_no[code] = -1
        
        probs_yes = self.model.predict(answers_yes)
        probs_no = self.model.predict(answers_no)

        entropy_yes = entropy(list(probs_yes.values()))
        entropy_no = entropy(list(probs_no.values()))

        p_yes = probs_yes.get(code, 0.5)
        p_no = 1 - p_yes
        expected_entropy = p_yes * entropy_yes + p_no * entropy_no

        return current_entropy - expected_entropy

    def record_answer(self, code, answer):
        self.answers[code] = answer

    def get_evidence_vector(self):
        return self.answers
