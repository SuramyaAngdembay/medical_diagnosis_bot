from llm_handler import GeminiAPI

def test_gemini_api():
    """Test the Gemini API integration."""
    print("Testing Gemini API integration...")
    
    try:
        # Initialize the Gemini API
        gemini = GeminiAPI()
        
        # Test a simple prompt
        prompt = "What are the common symptoms of the flu?"
        print(f"\nSending prompt: '{prompt}'")
        
        # Generate content
        response = gemini.generate_content(prompt)
        
        # Extract and print the text
        text = gemini.extract_text(response)
        print("\nResponse:")
        print(text)
        
        print("\nTest completed successfully!")
        return True
    
    except Exception as e:
        print(f"\nError: {e}")
        return False

if __name__ == "__main__":
    test_gemini_api() 