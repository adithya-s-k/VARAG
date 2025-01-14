import os
from typing import Optional, List, Union, Dict, Any
import warnings
import base64
from PIL import Image
import litellm
from abc import ABC, abstractmethod
from varag.llms import BaseLLM
from io import BytesIO


class LiteLLM:
    """
    A flexible wrapper for LLM models using LiteLLM as the backend.
    Supports both vision and text-only models with automatic validation.
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        is_vision_required: bool = False,
        verbose: Optional[bool] = False,
        **kwargs
    ):
        """
        Initialize the LLM wrapper.
        
        Args:
            model: The model identifier (e.g., "gpt-4", "claude-3-opus-20240229")
            api_key: Optional API key (will use environment variables if not provided)
            api_base: Optional API base URL
            is_vision_required: If True, will raise an error if the model doesn't support vision
            **kwargs: Additional arguments to pass to litellm
            
        Raises:
            ValueError: If is_vision_required is True but the model doesn't support vision
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.verbose = verbose
        self.kwargs = kwargs
        
        # Cache vision capability
        self._is_vision_model = litellm.supports_vision(model=self.model)
        
        # set verbosity of litellm by default - false
        litellm.set_verbose=self.verbose
        
        # Check vision requirement before proceeding
        if is_vision_required and not self._is_vision_model:
            raise ValueError(
                f"Model '{model}' does not support vision inputs, but vision capability was required. "
                "Please choose a vision-capable model."
            )
        
        # Validate environment and model access
        self._validate_setup()

    def _validate_setup(self) -> None:
        """Validate the environment setup and model access."""
        # Validate environment variables
        env_config = litellm.validate_environment(model=self.model)
        if not env_config["keys_in_environment"] and not self.api_key:
            raise ValueError(f"Missing API key for model {self.model}")
            
        # Validate model access
        if not litellm.check_valid_key(model=self.model, api_key=self.api_key):
            raise ValueError(f"Invalid API key for model {self.model}")

    def _prepare_messages(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        images: Optional[List[Union[str, Image.Image]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for the LLM API call.
        
        Args:
            query: The user query
            system_prompt: Optional system prompt
            context: Optional context to prepend to the query
            images: Optional list of images (for vision models)
            
        Returns:
            List of message dictionaries for the API call
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Prepare user message
        user_message = query
        if context:
            user_message = f"Context:\n{context}\n\nQuery:\n{query}"
            
        # Handle images for vision models
        if images and self._is_vision_model:
            content = []
            # Add text content
            content.append({"type": "text", "text": user_message})
            
            # Add images
            for img in images:
                if isinstance(img, str):
                    # Assume it's a base64 string or file path
                    if img.startswith(('http://', 'https://', 'data:')):
                        image_url = img
                    else:
                        with open(img, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                            image_url = f"data:image/jpeg;base64,{img_data}"
                else:
                    # Handle PIL Image
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_data = base64.b64encode(buffered.getvalue()).decode()
                    image_url = f"data:image/jpeg;base64,{img_data}"
                    
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
                
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_message})
            
        return messages

    async def aquery(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        images: Optional[List[Union[str, Image.Image]]] = None,
        **kwargs
    ) -> str:
        """
        Async query the model.
        
        Args:
            query: The query text
            system_prompt: Optional system prompt
            context: Optional context to prepend to the query
            images: Optional list of images (for vision models)
            **kwargs: Additional arguments to pass to litellm
            
        Returns:
            The model's response text
        """
        if images and not self._is_vision_model:
            raise ValueError(f"Model {self.model} does not support vision inputs")
            
        messages = self._prepare_messages(
            query=query,
            system_prompt=system_prompt,
            context=context,
            images=images
        )
        
        # Merge instance kwargs with method kwargs
        call_kwargs = {**self.kwargs, **kwargs}
        
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                api_base=self.api_base,
                **call_kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling LLM API: {str(e)}")

    def query(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        images: Optional[List[Union[str, Image.Image]]] = None,
        **kwargs
    ) -> str:
        """
        Synchronous version of aquery. Uses litellm's completion instead of acompletion.
        
        Args:
            query: The query text
            system_prompt: Optional system prompt
            context: Optional context to prepend to the query
            images: Optional list of images (for vision models)
            **kwargs: Additional arguments to pass to litellm
            
        Returns:
            The model's response text
        """
        if images and not self._is_vision_model:
            raise ValueError(f"Model {self.model} does not support vision inputs")
            
        messages = self._prepare_messages(
            query=query,
            system_prompt=system_prompt,
            context=context,
            images=images
        )
        
        # Merge instance kwargs with method kwargs
        call_kwargs = {**self.kwargs, **kwargs}
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                api_base=self.api_base,
                **call_kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling LLM API: {str(e)}")

    def is_vision_model(self) -> bool:
        """Check if the current model supports vision inputs."""
        return self._is_vision_model