"""
Ollama API Client

Handles communication with Ollama LLM server.
Supports streaming and non-streaming responses.
"""

import asyncio
import time
from typing import Optional, AsyncGenerator
from dataclasses import dataclass

import httpx
from loguru import logger

from config import Config


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    total_duration_ms: int
    prompt_eval_count: int
    eval_count: int
    tokens_per_second: float


class OllamaClient:
    """Async client for Ollama API."""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.ollama_base_url.rstrip("/")
        self.model = config.ollama_model
        self.timeout = config.ollama_timeout
    
    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> list[str]:
        """List available models."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Optional[LLMResponse]:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Creativity (0.0-1.0)
            max_tokens: Max response tokens
            
        Returns:
            LLMResponse or None if failed
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system:
            payload["system"] = system
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.debug(f"Sending request to {self.base_url}/api/generate")
                logger.debug(f"Model: {self.model}, Prompt length: {len(prompt)} chars")
                
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama returned status {response.status_code}: {response.text}")
                    return None
                
                data = response.json()
                
                elapsed = time.time() - start_time
                
                # Extract metrics
                total_duration = data.get("total_duration", 0) / 1_000_000  # ns to ms
                prompt_eval_count = data.get("prompt_eval_count", 0)
                eval_count = data.get("eval_count", 0)
                
                # Calculate tokens per second
                eval_duration = data.get("eval_duration", 1) / 1_000_000_000  # ns to s
                tps = eval_count / eval_duration if eval_duration > 0 else 0
                
                logger.info(f"LLM response received in {elapsed:.1f}s ({eval_count} tokens, {tps:.1f} t/s)")
                
                return LLMResponse(
                    content=data.get("response", ""),
                    model=data.get("model", self.model),
                    total_duration_ms=int(total_duration),
                    prompt_eval_count=prompt_eval_count,
                    eval_count=eval_count,
                    tokens_per_second=tps,
                )
                
        except httpx.TimeoutException:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return None
        except httpx.ConnectError as e:
            logger.error(f"Could not connect to Ollama at {self.base_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return None
    
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from LLM.
        
        Yields chunks of text as they arrive.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
        except Exception as e:
            logger.error(f"Streaming request failed: {e}")
            yield f"[ERROR: {e}]"
