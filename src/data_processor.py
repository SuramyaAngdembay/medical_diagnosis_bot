import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
import os

# Add parent directory to path to allow imports from rl_model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rl_model import (
    environment, NONE_VAL, PRES_VAL, ABS_VAL,
    encode_age, encode_sex, encode_race, encode_ethnicity
)


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


class DataProcessor:
    def __init__(self, evidence_path: str = "data/release_evidences.json", include_turns_in_state: bool = True):
        """Initialize the DataProcessor.
        
        Args:
            evidence_path: Path to the evidence metadata JSON file
            include_turns_in_state: Whether to include turn counter in state vector
        """
        self.logger = logging.getLogger(__name__)
        self.include_turns_in_state = include_turns_in_state
        
        # Load evidence metadata
        with open(evidence_path) as f:
            self.evidence_data = json.load(f)
            
        # Load conditions metadata
        with open("data/release_conditions.json") as f:
            self.conditions_data = json.load(f)
            
        # Store list of diseases
        self.diseases = list(self.conditions_data.keys())
            
        # Initialize symptom mappings
        self.symptom_name_2_index = {}  # Maps evidence codes to indices
        self.symptom_data_types = {}    # Maps evidence indices to data types
        self.symptom_possible_val_mapping = {}  # Maps evidence indices to possible values
        self.categorical_integer_symptoms = set()  # Set of indices for integer categorical symptoms
        self.symptom_default_value_mapping = {}  # Maps evidence indices to default values
        self.symptom_to_obs_mapping = {}  # Maps evidence indices to observation indices
        
        # Process evidence metadata
        start_obs_idx = 0
        for idx, (code, data) in enumerate(self.evidence_data.items()):
            self.symptom_name_2_index[code] = idx
            data_type = data.get("type", "B")  # Default to binary
            self.symptom_data_types[idx] = data_type
            
            if data_type in ["C", "M"]:
                possible_values = data.get("values", [])
                default_value = data.get("default_value", possible_values[0] if possible_values else None)
                self.symptom_default_value_mapping[idx] = default_value
                
                if data_type == "C" and all(v.isdigit() for v in possible_values):
                    # Integer categorical gets one dimension
                    self.categorical_integer_symptoms.add(idx)
                    self.symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + 1]
                    start_obs_idx += 1
                else:
                    # String categorical/multi-value gets one dimension per value
                    self.symptom_possible_val_mapping[idx] = {val: i for i, val in enumerate(possible_values)}
                    self.symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + len(possible_values)]
                    start_obs_idx += len(possible_values)
            else:
                # Binary evidence gets one dimension
                self.symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + 1]
                start_obs_idx += 1
                    
        # Calculate state dimensions
        self.symptom_size = start_obs_idx  # Total dimensions for evidence section
        
        # Set demographic feature dimensions (matching RL model exactly)
        self.num_age_values = 8    # Age categories [0-1, 1-4, 5-14, 15-29, 30-44, 45-59, 60-74, 75+]
        self.num_sex_values = 2    # Sex categories [M, F]
        self.num_race_values = 5   # Race categories [white, black, asian, native, other]
        self.num_ethnic_values = 2  # Ethnicity categories [hispanic, nonhispanic]
        
        # Calculate total state size
        self.context_size = (
            (1 if include_turns_in_state else 0) +  # Turn counter
            self.num_age_values +
            self.num_sex_values + 
            self.num_race_values +
            self.num_ethnic_values
        )
        self.state_size = self.symptom_size + self.context_size
        
        # Create action mask for symptoms (used in RL model)
        self.action_mask = np.zeros((len(self.symptom_name_2_index), self.state_size))
        for symptom_index in range(len(self.symptom_name_2_index)):
            start_idx = self.symptom_to_obs_mapping[symptom_index][0]
            end_idx = self.symptom_to_obs_mapping[symptom_index][1]
            self.action_mask[symptom_index, start_idx:end_idx] = 1
        
        self.logger.info(f"Initialized DataProcessor with state size {self.state_size}")
        self.logger.info(f"Evidence section size: {self.symptom_size}")
        self.logger.info(f"Context size: {self.context_size}")
        self.logger.info(f"Number of evidence codes: {len(self.symptom_name_2_index)}")
        self.logger.info(f"Number of multi-value evidences: {len([t for t in self.symptom_data_types.values() if t == 'M'])}")
        
    def get_state_from_answers(self, answers: Dict[str, any], demographics: Dict[str, any] = None) -> np.ndarray:
        """Convert the current answers and demographics into a state vector.
        
        Args:
            answers: Dictionary mapping evidence codes to their values
            demographics: Dictionary containing demographic information (age, sex, race, ethnicity)
            
        Returns:
            np.ndarray: State vector for the RL model
        """
        try:
            # Initialize state vector with NONE_VAL (0) for all dimensions
            state = np.ones(self.state_size, dtype=np.float32) * NONE_VAL
            
            # Process evidence answers
            for code, value in answers.items():
                if code not in self.symptom_name_2_index:
                    continue
                    
                symptom_index = self.symptom_name_2_index[code]
                start_idx, end_idx = self.symptom_to_obs_mapping[symptom_index]
                data_type = self.symptom_data_types[symptom_index]
                
                if data_type == "M":  # Multi-value evidence
                    # Initialize all values to ABS_VAL
                    state[start_idx:end_idx] = ABS_VAL
                    
                    if isinstance(value, str) and "@" in value:
                        _, selected_value = value.split("@")
                        selected_value = selected_value.strip()
                        
                        # Get the specific index for this value
                        if selected_value in self.symptom_possible_val_mapping[symptom_index]:
                            val_idx = self.symptom_possible_val_mapping[symptom_index][selected_value]
                            state[start_idx + val_idx] = PRES_VAL
                    else:
                        # If no valid value provided, use default if available
                        default_value = self.symptom_default_value_mapping.get(symptom_index)
                        if default_value is not None and default_value in self.symptom_possible_val_mapping[symptom_index]:
                            val_idx = self.symptom_possible_val_mapping[symptom_index][default_value]
                            state[start_idx + val_idx] = NONE_VAL
                            
                elif data_type == "C":  # Categorical evidence
                    if symptom_index in self.categorical_integer_symptoms:
                        # Handle integer categorical values
                        try:
                            val = float(value)
                            # Normalize to [0,1] range
                            num_vals = len(self.symptom_possible_val_mapping[symptom_index])
                            scaled = NONE_VAL + ((PRES_VAL - NONE_VAL) * val / num_vals)
                            state[start_idx] = scaled
                        except (ValueError, TypeError):
                            state[start_idx] = ABS_VAL
                    else:
                        # Handle string categorical values
                        state[start_idx:end_idx] = ABS_VAL
                        if isinstance(value, str):
                            if value in self.symptom_possible_val_mapping[symptom_index]:
                                val_idx = self.symptom_possible_val_mapping[symptom_index][value]
                                state[start_idx + val_idx] = PRES_VAL
                            elif self.symptom_default_value_mapping.get(symptom_index) == value:
                                val_idx = self.symptom_possible_val_mapping[symptom_index][value]
                                state[start_idx + val_idx] = NONE_VAL
                                
                else:  # Binary evidence
                    if isinstance(value, str):
                        state[start_idx] = PRES_VAL if value.lower() == "yes" else ABS_VAL
                    else:
                        try:
                            state[start_idx] = float(value)
                        except (ValueError, TypeError):
                            state[start_idx] = ABS_VAL
            
            # Set demographic features
            if demographics is not None:
                current_idx = self.symptom_size
                
                # 1. Turn counter (if used)
                if self.include_turns_in_state:
                    turn = demographics.get("turn", 0)
                    max_turns = demographics.get("max_turns", 20)
                    state[current_idx] = min(1.0, max(0.0, turn / max_turns))  # Clamp to [0,1]
                    current_idx += 1
                
                # 2. Age categories
                age_idx = encode_age(demographics.get("age", 30))
                state[current_idx + age_idx] = 1
                current_idx += self.num_age_values
                
                # 3. Sex categories
                sex_idx = encode_sex(demographics.get("sex", "M"))
                state[current_idx + sex_idx] = 1
                current_idx += self.num_sex_values
                
                # 4. Race categories
                race_idx = encode_race(demographics.get("race", "other"))
                state[current_idx + race_idx] = 1
                current_idx += self.num_race_values
                
                # 5. Ethnicity categories
                ethnic_idx = encode_ethnicity(demographics.get("ethnicity", "nonhispanic"))
                state[current_idx + ethnic_idx] = 1
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error creating state vector: {str(e)}")
            # Return state vector initialized with NONE_VAL
            return np.ones(self.state_size, dtype=np.float32) * NONE_VAL

    def get_state_size(self) -> int:
        """Get the size of the state vector."""
        return self.state_size
        
    def get_disease_count(self) -> int:
        """Get the number of possible diseases/conditions."""
        return len(self.diseases)
        
    def get_symptom_count(self) -> int:
        """Get the number of possible symptoms/evidences."""
        return len(self.symptom_name_2_index)
        
    def get_evidence_data(self, code: str) -> Dict:
        """Get metadata for a specific evidence code."""
        return self.evidence_data.get(code, {})
        
    def validate_evidence_structure(self, code: str, value: any) -> bool:
        """Validate that a given evidence value matches the expected structure."""
        evidence_data = self.get_evidence_data(code)
        if not evidence_data:
            return False
            
        data_type = evidence_data.get("data_type", "B")
        if data_type == "M":
            # Multi-value type
            if isinstance(value, str) and "@" in value:
                _, selected_value = value.split("@")
                selected_value = selected_value.strip()
                return selected_value in evidence_data.get("possible-values", [])
        else:
            # Binary/categorical
            if isinstance(value, str):
                return value.lower() in ["yes", "no"]
            return isinstance(value, (int, float))
            
        return False
