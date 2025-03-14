import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
from pydantic import BaseModel

from litellm import completion
from google import genai
from google.genai import types


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self,
                 messages: List[Dict[str, str]],
                 response_format: Optional[Type[BaseModel]] = None) -> str:
        """Generate LLM response from messages."""
        pass


class LiteLLMProvider(LLMProvider):
    """LiteLLM-based provider implementation."""

    def __init__(self, model_id: str):
        self.config = {"model": model_id}

    def generate(self,
                 messages: List[Dict[str, str]],
                 response_format: Optional[Type[BaseModel]] = None) -> str:
        """Generate using LiteLLM."""
        response = completion(
            **self.config,
            messages=messages,
            response_format=response_format
        )
        return response.choices[0].message.content


class GeminiProvider(LLMProvider):
    """Google's Gemini API provider implementation."""

    def __init__(self, model_id: str):
        self.model = model_id
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    def generate(self,
                 messages: List[Dict[str, str]],
                 response_format: Optional[Type[BaseModel]] = None) -> str:
        """Generate using Gemini."""
        # Convert messages to Gemini format
        system_msg = next((m["content"]
                          for m in messages if m["role"] == "system"), "")
        user_msg = next((m["content"]
                        for m in messages if m["role"] == "user"), "")

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_msg,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=response_format,
                system_instruction=system_msg
            )
        )
        return response.text
