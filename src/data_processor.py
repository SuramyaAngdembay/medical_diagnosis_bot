import json
import logging
from typing import Dict, List, Optional
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvidenceLoader:
    """A class to load and process evidence data from a JSON file.
    
    This class handles loading and accessing evidence data from a JSON file containing
    medical evidence information. It provides methods to access questions and possible
    values for each evidence code.
    """
    
    DEFAULT_EVIDENCE_PATH = "data/release_evidences.json"
    
    def __init__(self, evidence_path: Optional[str] = None) -> None:
        """Initialize the EvidenceLoader with a path to the evidence JSON file.
        
        Args:
            evidence_path (Optional[str]): Path to the JSON file containing evidence data.
                                         If None, uses the default path in the data directory.
            
        Raises:
            FileNotFoundError: If the evidence file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If the evidence data structure is invalid
        """
        # Use default path if none provided
        self.evidence_path = Path(evidence_path) if evidence_path else Path(self.DEFAULT_EVIDENCE_PATH)
        
        # Try to find the file relative to the current file's location
        if not self.evidence_path.exists():
            # Get the directory where this Python file is located
            current_dir = Path(__file__).parent.parent
            self.evidence_path = current_dir / self.DEFAULT_EVIDENCE_PATH
            
            if not self.evidence_path.exists():
                raise FileNotFoundError(
                    f"Evidence file not found at: {self.evidence_path}\n"
                    f"Please ensure the correct file path is provided or that the evidence file exists in {current_dir / self.DEFAULT_EVIDENCE_PATH}\n"
                    f"Current working directory: {Path.cwd()}"
                )
            
        try:
            with open(self.evidence_path, "r", encoding='utf-8') as f:
                self.evidences = json.load(f)
            logger.info(f"Successfully loaded {len(self.evidences)} evidence codes from {self.evidence_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {self.evidence_path}: {str(e)}")
            raise json.JSONDecodeError(f"Invalid JSON in evidence file: {e}", e.doc, e.pos)
            
        if not isinstance(self.evidences, dict):
            logger.error(f"Invalid data structure in {self.evidence_path}: expected dict, got {type(self.evidences)}")
            raise ValueError("Evidence data must be a dictionary")

    def get_all_evidence_codes(self) -> List[str]:
        """Get a list of all evidence codes in the dataset.
        
        Returns:
            List[str]: List of evidence codes
        """
        return list(self.evidences.keys())

    def get_question(self, code: str) -> str:
        """Get the question text for a given evidence code.
        
        Args:
            code (str): The evidence code to look up
            
        Returns:
            str: The question text in English, or a default message if not found
            
        Raises:
            KeyError: If the evidence code doesn't exist
        """
        if code not in self.evidences:
            logger.warning(f"Question requested for non-existent evidence code: '{code}'")
            raise KeyError(f"Evidence code '{code}' not found")
            
        return self.evidences[code].get("question_en", "No question available")

    def get_possible_values(self, code: str) -> List[str]:
        """Get the list of possible values for a given evidence code.
        
        Args:
            code (str): The evidence code to look up
            
        Returns:
            List[str]: List of possible values in English
            
        Raises:
            KeyError: If the evidence code doesn't exist
        """
        if code not in self.evidences:
            logger.warning(f"Possible values requested for non-existent evidence code: '{code}'")
            raise KeyError(f"Evidence code '{code}' not found")
            
        possible_values = self.evidences[code].get("value_meaning", {})
        if not possible_values:
            logger.debug(f"No value_meaning found for code '{code}', using default Yes/No values")
            return ["Yes", "No"]
            
        try:
            return [val["en"] for val in possible_values.values()]
        except (KeyError, TypeError) as e:
            logger.warning(f"Error processing value_meaning for code '{code}': {str(e)}, using default Yes/No values")
            return ["Yes", "No"]

    def get_evidence_data(self, code: str) -> Optional[Dict]:
        """Get the complete evidence data for a given code.
        
        Args:
            code (str): The evidence code to look up
            
        Returns:
            Optional[Dict]: The complete evidence data or None if not found
        """
        if code not in self.evidences:
            logger.warning(f"Evidence data requested for non-existent code: '{code}'")
            return None
        return self.evidences.get(code)

    def validate_evidence_structure(self) -> bool:
        """Validate the structure of the loaded evidence data.
        
        Returns:
            bool: True if the structure is valid, False otherwise
        """
        try:
            is_valid = all(
                isinstance(data, dict) and 
                "question_en" in data and 
                ("value_meaning" not in data or isinstance(data["value_meaning"], dict))
                for data in self.evidences.values()
            )
            if not is_valid:
                logger.error("Evidence data structure validation failed")
            return is_valid
        except Exception as e:
            logger.error(f"Error during evidence structure validation: {str(e)}")
            return False
