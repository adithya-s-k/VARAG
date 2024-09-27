import os
from typing import Optional, List, Dict
from openai import OpenAI


class BaseLLM:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = self._validate_model(model)
        self.client = self._initialize_client()

    def _initialize_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _validate_model(self, model: str) -> str:
        if model not in self.MODELS:
            raise ValueError(f"Invalid model. Choose from: {', '.join(self.MODELS)}")
        return model

    def query(
        self,
        context: str,
        system_prompt: str,
        query: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": query},
        ]
        return self._generate_response(messages, max_tokens)

    def chat(
        self,
        chat_history: List[Dict[str, str]],
        system_prompt: str,
        query: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = (
            [{"role": "system", "content": system_prompt}]
            + chat_history
            + [{"role": "user", "content": query}]
        )
        return self._generate_response(messages, max_tokens)

    def _generate_response(
        self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None
    ) -> str:
        completion_params = {
            "model": self.model,
            "messages": messages,
        }
        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**completion_params)
        return response.choices[0].message.content
