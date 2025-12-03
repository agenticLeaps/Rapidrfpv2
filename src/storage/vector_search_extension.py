#!/usr/bin/env python3
"""
Vector Search Extension for NeonDB Storage
"""

import psycopg2
import psycopg2.extras
import logging
from typing import List, Dict
from .neon_storage import NeonDBStorage

logger = logging.getLogger(__name__)

class VectorSearchMixin:
    """Mixin to add vector search capabilities to NeonDBStorage"""
    
    def vector_similarity_search_sync(self, 
                                      query_embedding: List[float], 
                                      org_id: str, 
                                      k: int = 10,
                                      similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Perform vector similarity search using PostgreSQL pgvector (synchronous)
        
        Args:
            query_embedding: Query vector as list of floats
            org_id: Organization ID to filter results
            k: Number of results to return
            similarity_threshold: Minimum cosine similarity threshold
            
        Returns:
            List of dictionaries with node_id, content, similarity_score, metadata
        """
        try:
            # Convert embedding to string format for PostgreSQL vector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            with self._get_sync_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            node_id,
                            node_type,
                            content,
                            graph_metadata,
                            1 - (embedding <=> %s::vector) as similarity_score
                        FROM noderag_embeddings 
                        WHERE org_id = %s 
                            AND 1 - (embedding <=> %s::vector) >= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (embedding_str, org_id, embedding_str, similarity_threshold, embedding_str, k))
                    
                    rows = cur.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            'node_id': row['node_id'],
                            'node_type': row['node_type'],
                            'content': row['content'],
                            'similarity_score': float(row['similarity_score']),
                            'metadata': row['graph_metadata'] or {}
                        })
                    
                    logger.debug(f"Vector search found {len(results)} results for org {org_id}")
                    return results
                    
        except Exception as e:
            logger.error(f"Error in vector similarity search for org {org_id}: {e}")
            return []

# Extend NeonDBStorage with vector search capabilities
class NeonDBStorageWithVector(NeonDBStorage, VectorSearchMixin):
    """NeonDBStorage with vector search capabilities"""
    pass