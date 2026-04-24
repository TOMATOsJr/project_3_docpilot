import os
import requests
from app.config import get_settings

settings = get_settings()
api_key = settings.gemini_api_key

response = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
if response.status_code == 200:
    models = response.json().get('models', [])
    for model in models:
        if 'embed' in model['name'].lower():
            print(f"Found embedding model: {model['name']} - Supported methods: {model.get('supportedGenerationMethods')}")
else:
    print(f"Failed to list models: {response.text}")
