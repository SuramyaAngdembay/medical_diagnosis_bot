import random


# Mock Model Hard coded for functionality testing
class MockModel:
    def predict(self, evidences):
        # Generate random probabilities for each condition
        conditions = ["Pneumonia", "Bronchitis", "Asthma", "COVID-19"]
        probabilities = {cond: random.uniform(0.1, 0.5) for cond in conditions}
        normalization_factor = sum(probabilities.values())
        return {cond: prob / normalization_factor for cond, prob in probabilities.items()}