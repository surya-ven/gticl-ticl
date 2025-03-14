import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
from pydantic import BaseModel

from litellm import completion
from google import genai
from google.genai import types

# Import openai for direct OpenAI access
from openai import OpenAI


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


class OpenAIProvider(LLMProvider):
    """Direct OpenAI API provider implementation."""

    def __init__(self, api_key: str, model_id: str, base_url: str = None):
        """
        Initialize OpenAI provider with model ID.

        Args:
            api_key: Provider API key
            model_id: The OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            base_url: Optional base URL for the OpenAI API
        """
        self.model = model_id
        self.base_url = base_url
        self.api_key = api_key
        # Initialize the OpenAI client with the provided API key and base URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self,
                 messages: List[Dict[str, str]],
                 response_format: Optional[Type[BaseModel]] = None) -> str:
        """
        Generate text using OpenAI API directly.

        Args:
            messages: List of message dictionaries with "role" and "content"
            response_format: Optional Pydantic model for structured output or response format dict

        Returns:
            Text response from the model or structured response
        """
        try:
            # Handle different types of response formats
            if response_format is None:
                # No specific format requested
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                )
                return response.choices[0].message.content

            elif isinstance(response_format, dict):
                # Direct JSON response format (OpenAI's native format)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    response_format=response_format
                )
                # Return the JSON content
                return response.choices[0].message.content

            elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Using regular completions with JSON response format
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )

                # Get JSON string from the response
                json_str = response.choices[0].message.content

                # For OpenRouter and other providers that don't support direct parsing
                if json_str:
                    import json
                    try:
                        # Try to parse as JSON first
                        parsed_json = json.loads(json_str)
                        # Return the parsed JSON as a dictionary
                        return parsed_json
                    except json.JSONDecodeError:
                        print(
                            f"Warning: Could not parse response as JSON: {json_str[:100]}...")
                        return json_str
                else:
                    return {}

            else:
                # Fallback for other types
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                )
                return response.choices[0].message.content

        except Exception as e:
            print(f"Error in OpenAI generation: {e}")
            # Return a simple error message that won't cause downstream errors
            return '{"rankings": []}'
