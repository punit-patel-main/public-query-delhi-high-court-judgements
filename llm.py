# llm.py
import random
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# List of API keys
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3")
]

GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]

class LLM:
    def __init__(self, provider=None, model=None, api_key=None):
        self.provider = provider.lower() if provider else random.choice(['groq', 'gemini'])
        self.model = model or self._default_model()
        self.api_key = api_key or self._random_api_key()
        self.api_key_name = self._get_api_key_name()

        self.system_prompt = ""
        self.user_prompt = ""

    def _random_api_key(self):
        if self.provider == 'groq':
            return random.choice([k for k in GROQ_API_KEYS if k])
        else:
            return random.choice([k for k in GEMINI_API_KEYS if k])

    def _get_api_key_name(self):
        """Get the environment variable name for the current API key"""
        if self.provider == 'groq':
            api_keys = GROQ_API_KEYS
            key_names = ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]
        else:
            api_keys = GEMINI_API_KEYS
            key_names = ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]
        
        # Find which key matches the selected API key
        for i, key in enumerate(api_keys):
            if key == self.api_key:
                return key_names[i]
        return "UNKNOWN_KEY"

    def _default_model(self):
        return "llama3-70b-8192" if self.provider == 'groq' else "gemini-1.5-flash-latest"

    def ask(self, system_prompt=None, user_prompt=None):
        self.system_prompt = system_prompt or self.system_prompt
        self.user_prompt = user_prompt or self.user_prompt

        if self.provider == 'groq':
            response_text = self._call_groq()
        elif self.provider == 'gemini':
            response_text = self._call_gemini()
        else:
            raise ValueError("Unsupported provider. Use 'groq' or 'gemini'.")
        
        print('\n\n')
        print('llm.ask() response:')
        print({
            'text': response_text,
            'provider': self.provider,
            'api_key_name': self.api_key_name,
            'model': self.model
        })
        print('\n\n')

        # Return dictionary with all requested information
        return {
            'text': response_text,
            'provider': self.provider,
            'api_key_name': self.api_key_name,
            'model': self.model
        }

    def _call_groq(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt}
            ]
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(f"Groq API request failed with status {response.status_code}: {response.text}")
        
        return response.json()['choices'][0]['message']['content']

    def _call_gemini(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

            model = genai.GenerativeModel(self.model)
            chat = model.start_chat()
            if self.system_prompt:
                chat.send_message(self.system_prompt)  # optional
            response = chat.send_message(self.user_prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API request failed: {str(e)}")