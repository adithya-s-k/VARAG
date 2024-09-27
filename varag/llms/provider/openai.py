import os
from typing import Optional, List, Union
import warnings
from PIL import Image
from openai import OpenAI
from varag.llms import BaseLLM


class OpenAILLM(BaseLLM):
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"

    MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4",
        "gpt-3.5-turbo",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(api_key, base_url or self.DEFAULT_BASE_URL, model)
