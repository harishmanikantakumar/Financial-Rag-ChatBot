import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Read the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# We'll enable this later after adding the API key.
# if OPENAI_API_KEY is None:
#     raise ValueError("OPENAI_API_KEY not found in .env file")