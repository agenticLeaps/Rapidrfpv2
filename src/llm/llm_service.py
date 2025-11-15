import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from dataclasses import dataclass
from openai import OpenAI
from gradio_client import Client

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
    def __init__(self, language: str = "english"):
        self.openai_client = None
        self.embedding_client = None
        self.prompt_manager = PromptManager(language=language)
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize OpenAI and HF embedding clients."""
        try:
            # Initialize OpenAI for LLM
            self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")
            
            # Initialize HF Gradio client for embeddings
            self.embedding_client = Client(Config.QWEN_EMBEDDING_ENDPOINT)
            logger.info(f"HF embedding client initialized: {Config.QWEN_EMBEDDING_ENDPOINT}")
                
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    def extract_semantic_units(self, text: str, max_units: int = 10) -> List[str]:
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
    
    def extract_entities(self, text: str, max_entities: int = 20) -> List[str]:
        """Extract named entities from text."""
        prompt = f"""Extract named entities from the following text. Focus on:
- People (names, titles, roles)
- Places (locations, buildings, geographical features)
- Organizations (companies, institutions, groups)
- Objects (specific items, products, concepts)
- Events (specific named events, meetings, projects)

Rules:
- Return only the entity names, not descriptions
- Use the most specific form (e.g., "Dr. John Smith" not just "John")
- Maximum {max_entities} entities
- Return as a JSON list of strings

Text: {text}

Entities:"""

        try:
            result = self._chat_completion(prompt, temperature=0.2)
            
            entities = self._parse_json_list(result)
            return entities[:max_entities]
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def extract_relationships(self, text: str, entities: List[str], max_relationships: int = 15) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities."""
        if len(entities) < 2:
            return []
        
        entities_str = ", ".join(entities)
        prompt = f"""Extract relationships between the given entities from the text. 

Entities: {entities_str}

Rules:
- Only use entities from the provided list
- Relationship format: (Entity1, Relationship, Entity2)
- Use clear, simple relationship terms (e.g., "works for", "located in", "created by")
- Maximum {max_relationships} relationships
- Return as JSON list of [entity1, relationship, entity2] arrays

Text: {text}

Relationships:"""

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
            
            # Make request with JSON mode if schema provided
            if json_schema:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            content = response.choices[0].message.content.strip()
            
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
            # Extract content from nodes
            content_pieces = []
            for node in community_nodes:
                if node.get('content'):
                    content_pieces.append(node['content'])
            
            combined_content = "\n".join(content_pieces[:10])  # Limit context
            
            # Use enhanced community summary prompt
            prompt = self.prompt_manager.community_summary.format(content=combined_content)
            
            result = self._chat_completion(prompt, temperature=0.6)
            return result.strip()
            
        except Exception as e:
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
        """Get embeddings for a list of texts using OpenAI embeddings."""
        if not texts:
            return []
        
        if batch_size is None:
            batch_size = min(Config.DEFAULT_BATCH_SIZE, 1000)  # OpenAI limit
        
        all_embeddings = []
        
        try:
            # Process in batches using OpenAI
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Use OpenAI embeddings API
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",  # Latest OpenAI embedding model
                    input=batch,
                    encoding_format="float"
                )
                
                # Extract embeddings from response
                embeddings = [data.embedding for data in response.data]
                
                if embeddings and len(embeddings) == len(batch):
                    all_embeddings.extend(embeddings)
                else:
                    logger.warning(f"Unexpected embedding result format for batch {i}")
                    # Add zero embeddings as fallback (1536 dimensions for OpenAI)
                    all_embeddings.extend([[0.0] * 1536 for _ in batch])
                
                # Small delay to avoid rate limiting
                if i + batch_size < len(texts):
                    time.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            # Return zero embeddings as fallback (1536 dimensions for OpenAI)
            return [[0.0] * 1536 for _ in texts]
    
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
        """Make OpenAI chat completion request."""
        try:
            if max_tokens is None:
                max_tokens = Config.DEFAULT_MAX_LENGTH
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
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