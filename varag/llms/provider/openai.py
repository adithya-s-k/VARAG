import os
from typing import Optional, List, Union
import warnings
from PIL import Image
from openai import OpenAI
from varag.llms import BaseLLM


class OpenAILLM(BaseLLM):
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_MAX_IMAGES = 5

    VISION_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "o1-mini",
        "o1-preview",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_images: int = DEFAULT_MAX_IMAGES,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model = self._validate_model(model)
        self.max_images = max_images
        self.client = self._initialize_client()

    def _validate_model(self, model: str) -> str:
        if model not in self.VISION_MODELS:
            warnings.warn(
                f"The model '{model}' is not in the list of known vision models. "
                f"It may not support image inputs or may not exist. "
                f"Known vision models are: {', '.join(self.VISION_MODELS)}",
                UserWarning,
            )
        return model

    def _initialize_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def query(
        self,
        query: str,
        context: str,
        system_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        context = f"""
        
        Given the following Context 
        
        Context
        =============================
         
        {context}
        
        =============================
        
        
        """

        completion_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(context + query)},
            ],
        }
        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**completion_params)
        return response.choices[0].message.content

    def __call__(self, image: Union[str, Image.Image], query: str) -> str:
        return self.response(query, image)
