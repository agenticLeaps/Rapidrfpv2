"""
AWS Bedrock client for Claude Sonnet and Cohere embeddings.
Replacement for Gemini/OpenAI/HuggingFace in Rapidrfpv2.
"""
import boto3
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class BedrockClient:
    """
    AWS Bedrock client for:
    - LLM: Claude Sonnet 4 (anthropic.claude-sonnet-4-20250514-v1:0)
    - Embeddings: Cohere embed-english-v3 (1024 dimensions)
    """

    def __init__(self, region: str = "us-east-1"):
        """Initialize Bedrock client."""
        self.region = region
        self.client = boto3.client("bedrock-runtime", region_name=region)
        # Use inference profile with us. prefix for on-demand throughput
        self.llm_model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        self.embedding_model = "cohere.embed-english-v3"
        self.embedding_dimension = 1024
        logger.info(f"Bedrock client initialized - Region: {region}, LLM: {self.llm_model}, Embeddings: {self.embedding_model}")

    def chat_completion(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using Claude Sonnet.

        Args:
            prompt: User prompt
            system: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        try:
            messages = [{"role": "user", "content": prompt}]

            # Note: Claude Sonnet 4.5 on Bedrock doesn't allow both temperature and top_p
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }

            if system:
                body["system"] = system

            response = self.client.invoke_model(
                modelId=self.llm_model,
                body=json.dumps(body)
            )

            result = json.loads(response["body"].read())
            return result["content"][0]["text"].strip()

        except Exception as e:
            logger.error(f"Bedrock chat completion error: {e}")
            raise

    def chat_completion_with_history(
        self,
        prompt: str,
        conversation_history: List[Dict[str, str]] = None,
        system: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text with conversation history and usage tracking.

        Args:
            prompt: Current user prompt
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            system: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with "response" and "usage" keys
        """
        try:
            # Build messages array
            messages = []

            # Add conversation history
            if conversation_history:
                for msg in conversation_history:
                    role = msg.get("role", "user")
                    # Claude uses "user" and "assistant" roles
                    if role not in ["user", "assistant"]:
                        role = "user"
                    messages.append({
                        "role": role,
                        "content": msg.get("content", "")
                    })

            # Add current prompt
            messages.append({"role": "user", "content": prompt})

            # Note: Claude Sonnet 4.5 on Bedrock doesn't allow both temperature and top_p
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }

            if system:
                body["system"] = system

            response = self.client.invoke_model(
                modelId=self.llm_model,
                body=json.dumps(body)
            )

            result = json.loads(response["body"].read())

            # Extract usage info
            usage = result.get("usage", {})
            usage_metadata = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            }

            return {
                "response": result["content"][0]["text"].strip(),
                "usage": usage_metadata
            }

        except Exception as e:
            logger.error(f"Bedrock chat completion with history error: {e}")
            raise

    def chat_completion_json(
        self,
        prompt: str,
        json_schema: Dict[str, Any] = None,
        system: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate JSON response using Claude Sonnet.

        Args:
            prompt: User prompt (should request JSON output)
            json_schema: Optional JSON schema for validation hints
            system: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Lower temperature for more deterministic JSON

        Returns:
            Parsed JSON dict
        """
        try:
            # Add JSON format instruction
            json_prompt = prompt
            if json_schema:
                json_prompt += "\n\nRespond with valid JSON only. No markdown, no explanation."

            text = self.chat_completion(
                prompt=json_prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Try direct parse
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Extract JSON from markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            # Try to find JSON object or array
            text = text.strip()
            if text.startswith("{") or text.startswith("["):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass

            # Last resort: find JSON in text
            import re
            json_match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            logger.error(f"Failed to parse JSON from Bedrock response: {text[:500]}")
            return {}

        except Exception as e:
            logger.error(f"Bedrock JSON completion error: {e}")
            raise

    def get_embeddings(
        self,
        texts: List[str],
        batch_size: int = 96,
        input_type: str = "search_document"
    ) -> List[List[float]]:
        """
        Get embeddings using Cohere embed-english-v3.

        Args:
            texts: List of texts to embed
            batch_size: Batch size (max 96 for Cohere)
            input_type: "search_document" for indexing, "search_query" for queries

        Returns:
            List of embedding vectors (1024 dimensions each)
        """
        if not texts:
            return []

        # Cohere has a 2048 character limit per text - truncate if needed
        MAX_TEXT_LENGTH = 2048
        truncated_texts = []
        for text in texts:
            if len(text) > MAX_TEXT_LENGTH:
                truncated_texts.append(text[:MAX_TEXT_LENGTH])
            else:
                truncated_texts.append(text)
        texts = truncated_texts

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        start_time = time.time()

        try:
            for i in range(0, len(texts), batch_size):
                batch_start = time.time()
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1

                # Cohere embedding request
                body = {
                    "texts": batch,
                    "input_type": input_type,
                    "truncate": "END"
                }

                response = self.client.invoke_model(
                    modelId=self.embedding_model,
                    body=json.dumps(body)
                )

                result = json.loads(response["body"].read())
                embeddings = result.get("embeddings", [])

                if embeddings and len(embeddings) == len(batch):
                    all_embeddings.extend(embeddings)
                    batch_time = time.time() - batch_start
                    logger.info(f"Embedding batch {batch_num}/{total_batches}: {len(batch)} items in {batch_time:.2f}s ({len(batch)/batch_time:.1f} items/sec)")
                else:
                    logger.warning(f"Unexpected embedding result for batch {batch_num}")
                    # Add zero embeddings as fallback
                    all_embeddings.extend([[0.0] * self.embedding_dimension for _ in batch])

            total_time = time.time() - start_time
            logger.info(f"Total embeddings: {len(all_embeddings)} in {total_time:.2f}s ({len(texts)/total_time:.1f} items/sec)")

            return all_embeddings

        except Exception as e:
            logger.error(f"Bedrock embeddings error: {e}")
            # Return zero embeddings on error
            return [[0.0] * self.embedding_dimension for _ in texts]

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a search query.
        Uses input_type="search_query" for better retrieval.

        Args:
            query: Search query text

        Returns:
            Embedding vector (1024 dimensions)
        """
        embeddings = self.get_embeddings([query], input_type="search_query")
        return embeddings[0] if embeddings else [0.0] * self.embedding_dimension
