# Import necessary modules and constants for use in other parts of the system
from .env import environment
from .sim_utils import encode_age, encode_sex, encode_race, encode_ethnicity
from .agent import Policy_Gradient_pair_model

# Define the constants needed by other modules
NONE_VAL = 0
PRES_VAL = 1
ABS_VAL = 2