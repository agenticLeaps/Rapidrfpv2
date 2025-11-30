import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import asyncio
from dataclasses import dataclass
from openai import AsyncOpenAI, OpenAI
from gradio_client import Client
import google.generativeai as genai
import os

from ..config.settings import Config
from .prompts import PromptManager

logger = logging.getLogger(__name__)

@dataclass
class EnhancedExtractionResult:
    """Enhanced extraction result matching NodeRAG's structured format."""
    semantic_units: List[Dict[str, Any]]  # Full semantic unit objects
    entities: List[str]
    relationships: List[Tuple[str, str, str]]  # (entity1, relation, entity2)
    success: bool
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

@dataclass
class ExtractionResult:
    """Backward compatibility for existing code."""
    semantic_units: List[str]
    entities: List[str]
    relationships: List[Tuple[str, str, str]]  # (entity1, relation, entity2)
    success: bool
    error_message: Optional[str] = None

class LLMService:
    def __init__(self, language: str = "english", model_type: str = "gemini"):
        self.openai_client = None
        self.async_openai_client = None
        self.embedding_client = None
        self.gemini_model = None
        self.model_type = model_type  # "openai", "gemini"
        self.prompt_manager = PromptManager(language=language)
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls_count = 0
        self.nodes_created = {
            'entities': 0,
            'relationships': 0, 
            'semantic_units': 0,
            'attributes': 0,
            'high_level': 0,
            'overview': 0,
            'text': 0
        }
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM clients based on model_type."""
        try:
            if self.model_type == "gemini":
                # Initialize Gemini 2.5 Flash Lite
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
                print("✅ Gemini 2.5 Flash Lite initialized")
                
            else:
                # Initialize OpenAI for LLM (both sync and async)
                self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                self.async_openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                print("✅ OpenAI clients initialized")
            
            # Initialize HF Gradio client for embeddings (always needed)
            self.embedding_client = Client(Config.QWEN_EMBEDDING_ENDPOINT)
            print("✅ HuggingFace embedding client initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    def extract_semantic_units(self, text: str, max_units: int = 3) -> List[str]:
        """Extract semantic units (independent events/ideas) from text."""
        prompt = f"""Extract independent semantic units from the following text. Each unit should be a complete, standalone concept or event that can be understood without additional context.

Rules:
- Each unit should be 1-2 sentences maximum
- Units should be independent and self-contained
- Focus on key events, facts, or ideas
- Maximum {max_units} units
- Return as a JSON list of strings

Text: {text}

Semantic Units:"""

        try:
            result = self._chat_completion(prompt, temperature=0.3)
            
            # Parse JSON response
            units = self._parse_json_list(result)
            return units[:max_units]
            
        except Exception as e:
            logger.error(f"Error extracting semantic units: {e}")
            return []
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text to avoid triggering Gemini safety filters."""
        # Remove potentially problematic patterns while preserving business content
        import re
        
        # Replace excessive capitalization that might look like shouting
        text = re.sub(r'\b[A-Z]{4,}\b', lambda m: m.group().capitalize(), text)
        
        # Limit text length to prevent overwhelming the model
        if len(text) > 8000:
            text = text[:8000] + "..."
            
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def extract_entities(self, text: str, max_entities: int = 20) -> List[str]:
        """Extract named entities from text."""
        print(f"   🔍 Extracting entities from text: {text[:100]}...")
        
        # Sanitize input text
        sanitized_text = self._sanitize_text(text)
        
        prompt = f"""Please analyze the following business text and identify key entities for knowledge extraction.

Focus on identifying:
- Person names and professional roles
- Company and organization names
- Location names
- Product or service names
- Project or initiative names

Guidelines:
- Extract only factual entity names mentioned in the text
- Use complete names when available
- Limit to {max_entities} most relevant entities
- Format as JSON array of strings

Business text to analyze:
{sanitized_text}

Please provide the entities in JSON format:"""

        try:
            print(f"   📤 Sending entity extraction prompt (length: {len(prompt)})")
            result = self._chat_completion(prompt, temperature=0.2)
            print(f"   📥 LLM response: {result[:200]}...")
            
            entities = self._parse_json_list(result)
            print(f"   ✅ Extracted {len(entities)} entities: {entities[:3]}")
            return entities[:max_entities]
            
        except Exception as e:
            print(f"   ❌ Entity extraction failed: {e}")
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def extract_relationships(self, text: str, entities: List[str], max_relationships: int = 15) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities."""
        if len(entities) < 2:
            return []
        
        # Sanitize input text
        sanitized_text = self._sanitize_text(text)
        
        entities_str = ", ".join(entities)
        prompt = f"""Please analyze the business text to identify relationships between the specified entities.

Available entities: {entities_str}

Instructions:
- Only connect entities from the provided list
- Use professional relationship terms (e.g., "employed by", "based in", "develops")
- Limit to {max_relationships} most relevant connections
- Format as JSON array with [entity1, relationship_type, entity2] structure

Business text for analysis:
{sanitized_text}

Please provide relationships in JSON format:"""

        try:
            result = self._chat_completion(prompt, temperature=0.2)
            
            relationships = self._parse_json_relationships(result)
            return relationships[:max_relationships]
            
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []
    
    def extract_all_from_chunk_enhanced(self, text: str) -> EnhancedExtractionResult:
        """
        Enhanced extraction using NodeRAG's unified structured approach.
        Extracts semantic units, entities, and relationships in a single LLM call.
        """
        try:
            # Use unified text decomposition prompt
            prompt = self.prompt_manager.text_decomposition.format(text=text)
            
            # Make LLM call with structured JSON response
            response = self._chat_completion_with_json(
                prompt, 
                json_schema=self.prompt_manager.text_decomposition_json,
                temperature=0.3
            )
            
            if not response or 'Output' not in response:
                return EnhancedExtractionResult(
                    semantic_units=[],
                    entities=[],
                    relationships=[],
                    success=False,
                    error_message="Invalid response format from LLM",
                    raw_response=response
                )
            
            # Parse structured response
            semantic_units = []
            all_entities = set()
            all_relationships = []
            
            for unit_data in response['Output']:
                # Validate unit structure
                if not all(key in unit_data for key in ['semantic_unit', 'entities', 'relationships']):
                    logger.warning(f"Incomplete semantic unit data: {unit_data}")
                    continue
                
                semantic_units.append(unit_data)
                
                # Collect unique entities (ensure UPPERCASE)
                for entity in unit_data.get('entities', []):
                    all_entities.add(entity.upper().strip())
                
                # Parse relationships
                for rel_str in unit_data.get('relationships', []):
                    relationship = self._parse_relationship_string(rel_str)
                    if relationship:
                        all_relationships.append(relationship)
            
            return EnhancedExtractionResult(
                semantic_units=semantic_units,
                entities=list(all_entities),
                relationships=all_relationships,
                success=True,
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced extraction: {e}")
            return EnhancedExtractionResult(
                semantic_units=[],
                entities=[],
                relationships=[],
                success=False,
                error_message=str(e)
            )
    
    def extract_all_from_chunk(self, text: str) -> ExtractionResult:
        """Extract all node types from a text chunk."""
        try:
            # Extract entities first as they're needed for relationships
            entities = self.extract_entities(text, Config.MAX_ENTITIES_PER_CHUNK)
            
            # Extract semantic units
            semantic_units = self.extract_semantic_units(text)
            
            # Extract relationships between entities
            relationships = self.extract_relationships(text, entities, Config.MAX_RELATIONSHIPS_PER_CHUNK)
            
            # Track nodes created
            self.track_nodes_created(
                entities=len(entities),
                relationships=len(relationships),
                semantic_units=len(semantic_units),
                text=1
            )
            
            return ExtractionResult(
                semantic_units=semantic_units,
                entities=entities,
                relationships=relationships,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in batch extraction: {e}")
            return ExtractionResult(
                semantic_units=[],
                entities=[],
                relationships=[],
                success=False,
                error_message=str(e)
            )
    
    def _parse_relationship_string(self, rel_str: str) -> Optional[Tuple[str, str, str]]:
        """Parse relationship string in format 'ENTITY_A, RELATION_TYPE, ENTITY_B'."""
        try:
            parts = [part.strip() for part in rel_str.split(',')]
            if len(parts) == 3:
                return (parts[0].upper(), parts[1], parts[2].upper())
            elif len(parts) > 3:
                # Handle cases where relation contains commas
                return (parts[0].upper(), ', '.join(parts[1:-1]), parts[-1].upper())
            else:
                logger.warning(f"Invalid relationship format: {rel_str}")
                return None
        except Exception as e:
            logger.error(f"Error parsing relationship string '{rel_str}': {e}")
            return None
    
    def _chat_completion_with_json(self, prompt: str, json_schema: Dict[str, Any] = None, temperature: float = 0.7, max_tokens: int = None) -> Dict[str, Any]:
        """Make OpenAI chat completion request with JSON schema validation."""
        try:
            if max_tokens is None:
                max_tokens = Config.DEFAULT_MAX_LENGTH
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # GPT-5-nano uses simplified responses.create API
            response = self.openai_client.responses.create(
                model="gpt-5-nano-2025-08-07",
                input=prompt
            )
            
            content = response.output_text.strip()
            
            # Parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {content}")
                return {}
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose query into main entities for search.
        Matches NodeRAG's query decomposition approach.
        """
        try:
            prompt = self.prompt_manager.query_decomposition.format(query=query)
            response = self._chat_completion_with_json(
                prompt, 
                json_schema=self.prompt_manager.query_decomposition_json,
                temperature=0.3
            )
            
            return response.get('elements', [])
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return []
    
    def reconstruct_relationship(self, malformed_relationship: List[str]) -> Tuple[str, str, str]:
        """
        Reconstruct malformed relationship triplet.
        Matches NodeRAG's relationship reconstruction.
        """
        try:
            prompt = self.prompt_manager.relationship_reconstruction.format(
                relationship=malformed_relationship
            )
            response = self._chat_completion_with_json(
                prompt,
                json_schema=self.prompt_manager.relationship_reconstruction_json,
                temperature=0.2
            )
            
            return (
                response.get('source', '').upper(),
                response.get('relationship', ''),
                response.get('target', '').upper()
            )
            
        except Exception as e:
            logger.error(f"Error reconstructing relationship: {e}")
            return ('', '', '')
    
    def generate_entity_attributes(self, entity: str, semantic_units: List[str], relationships: List[str]) -> str:
        """
        Generate comprehensive attributes for an important entity.
        Enhanced to match NodeRAG's structured input approach.
        """
        try:
            # Format semantic units and relationships for prompt
            semantic_units_text = "\n".join([f"- {unit}" for unit in semantic_units])
            relationships_text = "\n".join([f"- {rel}" for rel in relationships])
            
            prompt = self.prompt_manager.attribute_generation.format(
                entity=entity,
                semantic_units=semantic_units_text,
                relationships=relationships_text
            )
            
            result = self._chat_completion(prompt, temperature=0.5, max_tokens=2000)
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error generating entity attributes: {e}")
            return f"Error generating attributes for {entity}"
    
    def generate_entity_attributes_legacy(self, entity: str, context_chunks: List[str]) -> str:
        """Legacy method for backward compatibility."""
        context = "\n\n".join(context_chunks)
        
        prompt = f"""Generate comprehensive attributes for the entity "{entity}" based on the provided context. Include:

- Key characteristics and properties
- Role and importance in the context
- Relationships with other entities
- Notable actions or events involving this entity
- Any other relevant information

Context:
{context}

Generate a detailed but concise summary (2-3 paragraphs) about {entity}:"""

        try:
            result = self._chat_completion(prompt, temperature=0.5)
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error generating entity attributes: {e}")
            return f"Error generating attributes for {entity}"
    
    def generate_community_summary(self, community_nodes: List[Dict[str, Any]]) -> str:
        """
        Generate a high-level summary for a community of nodes.
        Enhanced to match NodeRAG's community summary approach.
        """
        try:
            print(f"   🔍 Generating community summary for {len(community_nodes)} nodes")
            
            # Extract content from nodes
            content_pieces = []
            for node in community_nodes:
                if node.get('content'):
                    content_pieces.append(node['content'])
            
            combined_content = "\n".join(content_pieces[:10])  # Limit context
            print(f"   📝 Combined content length: {len(combined_content)}")
            print(f"   📝 Content preview: {combined_content[:200]}...")
            
            # Use enhanced community summary prompt
            prompt = self.prompt_manager.community_summary.format(content=combined_content)
            print(f"   📤 Sending community summary prompt (length: {len(prompt)})")
            
            result = self._chat_completion(prompt, temperature=0.6)
            print(f"   📥 Community summary response: {result[:200]}...")
            
            return result.strip()
            
        except Exception as e:
            print(f"   ❌ Community summary generation failed: {e}")
            logger.error(f"Error generating community summary: {e}")
            return "Error generating community summary"
    
    def generate_community_overview(self, community_summary: str) -> str:
        """Generate a short keyword-based title for a community."""
        prompt = f"""Create a short, keyword-based title (3-8 words) that captures the main theme of this summary:

Summary: {community_summary}

Title:"""

        try:
            result = self._chat_completion(prompt, temperature=0.3, max_tokens=100)
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error generating community overview: {e}")
            return "Community Overview"
    
    def get_embeddings(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI embeddings with adaptive batching."""
        if not texts:
            return []
        
        if batch_size is None:
            batch_size = Config.get_adaptive_batch_size(len(texts))
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        try:
            # Process in batches using OpenAI with adaptive delays
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                current_batch_num = i // batch_size + 1
                
                # Get embeddings for this batch with retries
                batch_embeddings = self._get_batch_embeddings_with_retry(
                    batch, current_batch_num, total_batches
                )
                
                if batch_embeddings and len(batch_embeddings) == len(batch):
                    all_embeddings.extend(batch_embeddings)
                    print(f"   ✓ Batch {current_batch_num}/{total_batches} completed ({len(batch_embeddings)} embeddings)")
                else:
                    logger.warning(f"Batch {current_batch_num} failed, using fallback embeddings")
                    # Add zero embeddings as fallback (1536 dimensions for OpenAI)
                    all_embeddings.extend([[0.0] * 1536 for _ in batch])
                
                # Memory management - force garbage collection periodically
                if current_batch_num % Config.GC_EVERY_N_BATCHES == 0:
                    import gc
                    gc.collect()
                    if Config.IS_RENDER:
                        print(f"   🧹 Memory cleanup after batch {current_batch_num}")
                
                # Adaptive delay based on environment and progress
                if i + batch_size < len(texts):
                    delay = Config.get_adaptive_delay(current_batch_num, total_batches)
                    time.sleep(delay)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            # Return zero embeddings as fallback (1536 dimensions for OpenAI)
            return [[0.0] * 1536 for _ in texts]
    
    def _get_batch_embeddings_with_retry(self, batch: List[str], batch_num: int, total_batches: int) -> List[List[float]]:
        """Get embeddings for a batch with retry logic and rate limiting protection."""
        import openai
        from requests.exceptions import RequestException, Timeout, ConnectionError
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                # Use OpenAI embeddings API with timeout
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                    encoding_format="float",
                    timeout=60  # 60 second timeout
                )
                
                # Extract embeddings from response
                embeddings = [data.embedding for data in response.data]
                
                if embeddings and len(embeddings) == len(batch):
                    return embeddings
                else:
                    logger.warning(f"Batch {batch_num}: Unexpected embedding result format on attempt {attempt + 1}")
                    
            except openai.RateLimitError as e:
                wait_time = Config.RETRY_DELAY_BASE * (2 ** attempt)  # Exponential backoff
                if Config.IS_RENDER:
                    wait_time *= 2  # Double wait time for Render
                
                print(f"   ⚠️  Batch {batch_num}: Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{Config.MAX_RETRIES}")
                time.sleep(wait_time)
                continue
                
            except (openai.APITimeoutError, Timeout) as e:
                wait_time = Config.RETRY_DELAY_BASE * (attempt + 1)
                print(f"   ⏱️  Batch {batch_num}: Timeout on attempt {attempt + 1}, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                continue
                
            except (openai.APIConnectionError, ConnectionError) as e:
                wait_time = Config.RETRY_DELAY_BASE * (attempt + 1)
                print(f"   🔗 Batch {batch_num}: Connection error on attempt {attempt + 1}, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                continue
                
            except openai.BadRequestError as e:
                logger.error(f"Batch {batch_num}: Bad request error: {e}")
                # Don't retry bad requests
                break
                
            except Exception as e:
                wait_time = Config.RETRY_DELAY_BASE * (attempt + 1)
                print(f"   ❌ Batch {batch_num}: Unexpected error on attempt {attempt + 1}: {type(e).__name__}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(wait_time)
                continue
        
        # All retries failed
        logger.error(f"Batch {batch_num}: All {Config.MAX_RETRIES} attempts failed")
        return None
    
    def _parse_json_list(self, response: str) -> List[str]:
        """Parse JSON list from LLM response."""
        try:
            # Try to find JSON in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                # Fallback: split by lines and clean
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                return [line.lstrip('- ').strip('"') for line in lines if line]
            
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
            else:
                return []
                
        except json.JSONDecodeError:
            # Fallback parsing
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            return [line.lstrip('- ').strip('"') for line in lines if line]
    
    def _chat_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = None) -> str:
        """Make LLM completion request (Gemini or OpenAI)."""
        try:
            if max_tokens is None:
                max_tokens = Config.DEFAULT_MAX_LENGTH
            
            if self.model_type == "gemini":
                print(f"      🌐 Making API call to Gemini 2.5 Flash Lite")
                print(f"      📊 Parameters: temp={temperature}, max_tokens={max_tokens}")
                
                # Rate limiting for Gemini API (Flash Lite - Paid Tier 1: 4000 RPM)
                import time
                time.sleep(0.02)  # Minimal delay for 4000 RPM (about 0.015s between requests)
                
                # Configure generation settings with relaxed safety
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                
                # Configure safety settings to be less restrictive for technical content
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                ]
                
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # Check if response was blocked by safety filters
                if not response.parts:
                    finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                    print(f"      ⚠️  Response blocked by Gemini safety filters (finish_reason: {finish_reason})")
                    
                    # Provide more intelligent fallbacks based on prompt type
                    if "extract entities" in prompt.lower() or "entities" in prompt.lower():
                        content = "[]"  # Empty JSON array for entity extraction
                        print("      💡 Returning empty entities array due to safety filtering")
                    elif "relationships" in prompt.lower() or "extract relationships" in prompt.lower():
                        content = "[]"  # Empty JSON array for relationships  
                        print("      💡 Returning empty relationships array due to safety filtering")
                    elif "summary" in prompt.lower() or "summarize" in prompt.lower():
                        content = "Content summary unavailable due to safety filtering."
                        print("      💡 Returning safe summary placeholder due to safety filtering")
                    elif "decompos" in prompt.lower():
                        content = "[]"  # Empty array for decomposition
                        print("      💡 Returning empty decomposition array due to safety filtering")
                    else:
                        content = "Content filtered for safety."
                        print("      💡 Returning generic safe response due to safety filtering")
                else:
                    content = response.text.strip()
                
                # Track tokens (Gemini provides usage metadata)
                self.api_calls_count += 1
                if hasattr(response, 'usage_metadata'):
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    print(f"      📊 Tokens: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total")
                else:
                    # Estimate tokens if not available
                    estimated_input = len(prompt.split()) * 1.3  # Rough estimate
                    estimated_output = len(content.split()) * 1.3
                    self.total_input_tokens += int(estimated_input)
                    self.total_output_tokens += int(estimated_output)
                    print(f"      📊 Estimated tokens: {int(estimated_input)} input + {int(estimated_output)} output")
                
                print(f"      ✅ Gemini response received (length: {len(content)})")
                
                if not content:
                    print(f"      ⚠️  Empty response from Gemini!")
                    
                return content
                
            else:
                # OpenAI fallback
                print(f"      🌐 Making API call to gpt-5-nano-2025-08-07")
                print(f"      📊 Using default parameters (temperature=1.0)")
                
                response = self.openai_client.responses.create(
                    model="gpt-5-nano-2025-08-07",
                    input=prompt
                )
                
                content = response.output_text.strip()
                print(f"      ✅ API response received (length: {len(content)})")
                
                if not content:
                    print(f"      ⚠️  Empty response from API!")
                    
                return content
            
        except Exception as e:
            error_msg = str(e)
            print(f"      ❌ LLM API error: {error_msg}")
            logger.error(f"LLM API error: {error_msg}")
            
            # Handle rate limiting (429 errors) - optimized for Flash Lite Paid Tier 1
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                print("      ⏱️  Rate limit hit - adding brief delay for Paid Tier...")
                import time
                time.sleep(2)  # Brief 2-second delay for paid tier recovery
                print("      ✅ Resuming after rate limit delay")
                return ""  # Return empty to continue pipeline
            
            # Handle specific Gemini API errors gracefully
            elif "response.text" in error_msg and "finish_reason" in error_msg:
                print(f"      💡 Gemini response was blocked - returning safe fallback")
                if "summary" in prompt.lower():
                    return "Unable to generate summary due to content filtering."
                elif "entities" in prompt.lower():
                    return "[]"
                elif "relationships" in prompt.lower():
                    return "[]"
                else:
                    return "Content blocked by safety filters."
            else:
                raise
    
    def _parse_json_relationships(self, response: str) -> List[Tuple[str, str, str]]:
        """Parse JSON relationships from LLM response."""
        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                return []
            
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            relationships = []
            for item in parsed:
                if isinstance(item, list) and len(item) == 3:
                    relationships.append((str(item[0]), str(item[1]), str(item[2])))
            
            return relationships
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse relationships JSON")
            return []
    
    async def extract_all_from_chunks_batch(self, chunks: List[Any]) -> List[EnhancedExtractionResult]:
        """Extract from multiple chunks using proven individual processing approach in parallel."""
        if not chunks:
            return []
        
        print(f"   🔄 Processing batch of {len(chunks)} chunks in parallel...")
        
        async def process_single_chunk_async(chunk):
            """Process a single chunk using the proven individual approach"""
            try:
                # Use the same proven approach as extract_all_from_chunk but async
                text = chunk.content
                
                # Extract entities first as they're needed for relationships
                entities = self.extract_entities(text, Config.MAX_ENTITIES_PER_CHUNK)
                
                # Extract semantic units
                semantic_units = self.extract_semantic_units(text)
                
                # Extract relationships between entities
                relationships = self.extract_relationships(text, entities, Config.MAX_RELATIONSHIPS_PER_CHUNK)
                
                # Convert to EnhancedExtractionResult format for compatibility
                enhanced_result = EnhancedExtractionResult(
                    semantic_units=[{"semantic_unit": unit, "entities": entities, "relationships": [f"{rel[0]}, {rel[1]}, {rel[2]}" for rel in relationships]} for unit in semantic_units],
                    entities=entities,
                    relationships=relationships,
                    success=True
                )
                return enhanced_result
                
            except Exception as e:
                print(f"   ❌ Chunk processing failed: {e}")
                return EnhancedExtractionResult(
                    semantic_units=[], entities=[], relationships=[], 
                    success=False, error_message=str(e)
                )
        
        # Process all chunks in parallel using asyncio.gather
        try:
            # Convert sync calls to async using run_in_executor
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Create tasks for parallel processing
            tasks = []
            for chunk in chunks:
                task = loop.run_in_executor(None, lambda c=chunk: self._process_chunk_sync(c))
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert any exceptions to failed results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(EnhancedExtractionResult(
                        semantic_units=[], entities=[], relationships=[], 
                        success=False, error_message=str(result)
                    ))
                else:
                    final_results.append(result)
            
            return final_results
            
        except Exception as e:
            print(f"   ❌ Parallel batch processing failed: {e}")
            return [EnhancedExtractionResult(
                semantic_units=[], entities=[], relationships=[], success=False, error_message=str(e)
            ) for _ in chunks]
    
    def _process_chunk_sync(self, chunk) -> EnhancedExtractionResult:
        """Synchronous chunk processing using proven approach"""
        try:
            text = chunk.content
            
            # Use the proven individual extraction approach
            entities = self.extract_entities(text, Config.MAX_ENTITIES_PER_CHUNK)
            semantic_units = self.extract_semantic_units(text)
            relationships = self.extract_relationships(text, entities, Config.MAX_RELATIONSHIPS_PER_CHUNK)
            
            # Convert to EnhancedExtractionResult format
            return EnhancedExtractionResult(
                semantic_units=[{"semantic_unit": unit, "entities": entities, "relationships": [f"{rel[0]}, {rel[1]}, {rel[2]}" for rel in relationships]} for unit in semantic_units],
                entities=entities,
                relationships=relationships,
                success=True
            )
            
        except Exception as e:
            return EnhancedExtractionResult(
                semantic_units=[], entities=[], relationships=[], 
                success=False, error_message=str(e)
            )
    
    async def _async_chat_completion_with_json(self, prompt: str, json_schema: Dict[str, Any] = None, temperature: float = 0.7, max_tokens: int = None) -> Dict[str, Any]:
        """Async OpenAI chat completion with JSON response."""
        try:
            if max_tokens is None:
                max_tokens = Config.DEFAULT_MAX_LENGTH
            
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.async_openai_client.responses.create(
                model="gpt-5-nano-2025-08-07",
                input=prompt
            )
            
            content = response.output_text.strip()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse async JSON response: {content}")
                return {}
            
        except Exception as e:
            logger.error(f"Async OpenAI API error: {e}")
            raise
    
    async def generate_entity_attributes_batch(self, entities_with_context: List[Tuple[str, List[str], List[str]]]) -> List[str]:
        """Generate attributes for multiple entities in batch."""
        if not entities_with_context:
            return []
        
        try:
            # Create batch prompt
            batch_prompt = "Generate comprehensive attributes for the following entities:\n\n"
            
            for i, (entity, semantic_units, relationships) in enumerate(entities_with_context):
                batch_prompt += f"=== ENTITY {i+1}: {entity} ===\n"
                batch_prompt += f"Context: {'; '.join(semantic_units[:3])}\n"
                batch_prompt += f"Relationships: {'; '.join(relationships[:3])}\n\n"
            
            batch_prompt += "Return a JSON array with one attribute description per entity in the same order."
            
            # Use individual processing with ThreadPoolExecutor for parallel execution
            import concurrent.futures
            
            def generate_single_attribute(entity_data):
                entity, semantic_units, relationships = entity_data
                return self.generate_entity_attributes(entity, semantic_units, relationships)
            
            # Process in parallel using ThreadPoolExecutor (optimal for GPT-5-nano)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(generate_single_attribute, entity_data) for entity_data in entities_with_context]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            return results
                
        except Exception as e:
            logger.error(f"Error in batch attribute generation: {e}")
            return [f"Error generating attributes for entity" for _ in entities_with_context]
    
    def track_nodes_created(self, **node_counts):
        """Track the number of nodes created by type"""
        for node_type, count in node_counts.items():
            if node_type in self.nodes_created:
                self.nodes_created[node_type] += count
    
    def get_usage_stats(self):
        """Get comprehensive usage statistics"""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        total_nodes = sum(self.nodes_created.values())
        
        return {
            'api_calls': self.api_calls_count,
            'total_tokens': total_tokens,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_nodes': total_nodes,
            'nodes_by_type': self.nodes_created.copy(),
            'avg_tokens_per_call': round(total_tokens / max(self.api_calls_count, 1), 2),
            'model_type': self.model_type
        }
    
    def print_usage_summary(self):
        """Print a formatted usage summary"""
        stats = self.get_usage_stats()
        
        print("\n" + "="*60)
        print("📊 LLM USAGE STATISTICS")
        print("="*60)
        print(f"🤖 Model: {stats['model_type'].upper()}")
        print(f"📞 API Calls: {stats['api_calls']}")
        print(f"🔢 Total Tokens: {stats['total_tokens']:,}")
        print(f"   • Input Tokens: {stats['input_tokens']:,}")
        print(f"   • Output Tokens: {stats['output_tokens']:,}")
        print(f"📈 Avg Tokens/Call: {stats['avg_tokens_per_call']}")
        
        print(f"\n🕸️ NODES CREATED: {stats['total_nodes']}")
        for node_type, count in stats['nodes_by_type'].items():
            if count > 0:
                print(f"   • {node_type.capitalize()}: {count}")
        
        print("="*60)