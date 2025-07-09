import os
from dotenv import load_dotenv

# Clear any existing environment variables
if 'OPENAI_API_KEY' in os.environ:
    print(f"Found existing OPENAI_API_KEY in environment: {os.environ['OPENAI_API_KEY'][:10]}...")
    del os.environ['OPENAI_API_KEY']

# Load .env file
load_dotenv()

# Check what was loaded
api_key = os.getenv("OPENAI_API_KEY")
print(f"API key from .env: {api_key[:10]}...{api_key[-4:] if api_key else 'None'}")

# Also check if there's a .env.example file
if os.path.exists('.env.example'):
    print("⚠️  .env.example file exists - make sure you're not accidentally using it")
    
# Read .env file directly
print("\n--- Contents of .env file ---")
try:
    with open('.env', 'r') as f:
        for line_num, line in enumerate(f, 1):
            if 'OPENAI_API_KEY' in line:
                print(f"Line {line_num}: {line.strip()}")
except Exception as e:
    print(f"Error reading .env file: {e}")