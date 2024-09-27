import os
from typing import Optional, List, Union
from PIL import Image
from openai import OpenAI
from varag.llms import BaseLLM


class GroqLLM(BaseLLM):
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llava-v1.5-7b-4096-preview"
    DEFAULT_MAX_IMAGES = 5

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_images: int = DEFAULT_MAX_IMAGES,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model = model
        self.max_images = max_images
        self.client = self._initialize_client()

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
