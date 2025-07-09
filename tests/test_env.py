import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")  # Shows first 10 and last 4 chars
    print(f"Length: {len(api_key)} characters")
else:
    print("❌ No API key found")

print(f"Current working directory: {os.getcwd()}")
print(f"Looking for .env file at: {os.path.join(os.getcwd(), '.env')}")
print(f".env file exists: {os.path.exists('.env')}")