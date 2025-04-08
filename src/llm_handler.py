"""
LLM Handler

Handles interactions with the Gemini API.
"""

import os
import json
import requests
from dotenv import load_dotenv
import google.generativeai as genai

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from the .env file in the project root
load_dotenv(os.path.join(project_root, ".env"))

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiAPI:
    """Wrapper for the Gemini API."""
    
    def __init__(self, model_name="gemini-2.5-pro-exp-03-25"):
        """Initialize the Gemini API wrapper."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    def generate_content(self, prompt, max_tokens=1000):
        """Generate content using the Gemini API."""
        url = f"{self.base_url}?key={self.api_key}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        return response.json()
    
    def extract_text(self, response):
        """Extract the generated text from the API response."""
        try:
            return response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Error extracting text from response"


# Example usage
if __name__ == "__main__":
    gemini = GeminiAPI()
    response = gemini.generate_content("Explain how AI works")
    print(gemini.extract_text(response)) 