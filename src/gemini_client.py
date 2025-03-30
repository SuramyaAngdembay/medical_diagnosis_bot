import os
from dotenv import load_dotenv
import google.generativeai as genai

class GeminiClient:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in your environment.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

    def generate_response(self, question, user_response, options):
        print("Gemini API Called")
        print(f"Question: {question}")
        print(f"User Response: {user_response}")
        print(f"Options: {options}")
        formatted_prompt = f"""
        The user was asked: "{question}"
        The user responded: "{user_response}"

        Your task is to analyze the user's response within a medical context and select the single most appropriate answer from the provided options:
        {options}

        You have the skill to understand natural language and map it to a predefined set of options. Consider the medical context of the question and response.

        - If the response directly matches one of the provided options, return the matching option.
        - If the response indirectly implies one of the provided options (a symptom, condition, or history strongly associated with a specific category), return the best matching option.
        - If the response is uncertain, ambiguous, or unclear but leans towards one of the provided options, return the most likely choice.
        - If the response provides information that clearly falls within one of the categorical options (even if not an exact match), return that option.
        - Return 'NO_MATCH' only if the user's response has absolutely no bearing on the question asked or is completely unintelligible, or if it clearly contradicts all provided options without offering a plausible alternative within the question's domain.

        Return only one result — either one of the provided options or "NO_MATCH" — without any additional text or explanation.
        """
        print("--- Formatted Prompt Sent to Gemini ---")
        print(formatted_prompt)
        print("--- End of Prompt ---")

        try:
            response = self.model.generate_content(formatted_prompt)
            print(f"Gemini API Response: {response.text.strip()}")
            return response.text.strip()
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            return "NO_MATCH"

if __name__ == "__main__":
    load_dotenv()  # Load your API key if you haven't already
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found. Please set it in your environment or .env file.")
    else:
        genai.configure(api_key=api_key)
        print("--- Available Gemini Models Supporting generateContent ---")
        found_gemini_pro = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                if m.name == 'models/gemini-pro':
                    found_gemini_pro = True
        if found_gemini_pro:
            print("\n'models/gemini-pro' is available and supports generateContent.")
        else:
            print("\n'models/gemini-pro' was NOT found in the list of available models supporting generateContent.")