"""
OpenRoute LLM integration for Arabic RAG system.
Provides a clean wrapper for the OpenRouter API using Mistral or similar LLMs.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# Load environment variables
load_dotenv()


class OpenRouteLLM(LLM):
    """LangChain-compatible LLM wrapper for OpenRouter API."""

    api_key: str = os.getenv("OPENROUTE_API_KEY")
    model_name: str = "mistralai/mistral-7b-instruct"
    temperature: float = 0.7
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        """Identify LLM type for LangChain integration."""
        return "openroute"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the OpenRouter API with a given prompt.

        Args:
            prompt: User or system prompt to send.
            stop: Optional stop sequences.
            run_manager: LangChain callback manager (unused).
        Returns:
            Generated text (Arabic or English depending on prompt).
        """
        if not self.api_key:
            raise ValueError("❌ Missing OPENROUTE_API_KEY in environment variables.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if stop:
            payload["stop"] = stop

        print(f"[INFO] Sending request to OpenRouter model: {self.model_name}")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
        )

        # Handle request errors
        if response.status_code != 200:
            print(f"[ERROR] OpenRouter API response: {response.text}")
            raise ValueError(f"OpenRouter API Error ({response.status_code}): {response.text}")

        response_data = response.json()

        try:
            answer = response_data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            print(f"[ERROR] Unexpected response structure: {response_data}")
            answer = "عذرًا، لم أتمكن من توليد إجابة حالياً."

        return answer

    def get_num_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        Useful for controlling input length in RAG pipelines.
        """
        # Rough approximation: 4 chars per token for Arabic and English
        return len(text) // 4
