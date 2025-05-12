"""
Custom LLM implementation for the PowerBI Agent.
"""

import json
import urllib.request
from langchain.llms.base import LLM

GPT_API_KEY = "2b24fef721d14c94a333ab2e4f686f40"

class CustomGPT(LLM):
    def _call(self, prompt: str, model="gpt-4o", version="2024-02-01", **kwargs):
        url = f"https://apigatewayinnovation.azure-api.net/openai-api/deployments/{model}/chat/completions?api-version={version}"
        headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": GPT_API_KEY,
        }
        data = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            "stop": ["\nThought:", "\nObservation:"],
        }
        try:
            request = urllib.request.Request(
                url, headers=headers, data=json.dumps(data).encode("utf-8")
            )
            request.get_method = lambda: "POST"
            response = urllib.request.urlopen(request)
            response_data = json.loads(response.read())
            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Erreur lors de l'appel à l'API : {e}"

    @property
    def _llm_type(self) -> str:
        return "customGPT"