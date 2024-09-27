import os
from typing import Optional, List, Union, Dict
import warnings
from PIL import Image
from openai import OpenAI
from varag.llms import BaseLLM


class GroqLLM(BaseLLM):
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "mixtral-8x7b-32768"

    MODELS = [
        "gemma2-9b-it",
        "gemma-7b-it",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-11b-text-preview",
        "llama-3.2-90b-text-preview",
        "llama-guard-3-8b",
        "llava-v1.5-7b-4096-preview",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        super().__init__(api_key, base_url or self.DEFAULT_BASE_URL, model)

    def list_models(self) -> List[Dict]:
        url = f"{self.base_url}/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self.client.get(url, headers=headers)
        return response.json()
