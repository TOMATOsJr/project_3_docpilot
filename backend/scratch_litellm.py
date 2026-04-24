import litellm
import os
from app.config import get_settings

settings = get_settings()
os.environ["GEMINI_API_KEY"] = settings.gemini_api_key

try:
    response = litellm.embedding(model="gemini/gemini-embedding-2", input=["hello world"])
    print("Embedding generated successfully with gemini/gemini-embedding-2!")
    print(len(response.data[0]['embedding']))
except Exception as e:
    print(f"gemini/gemini-embedding-2 failed: {e}")
