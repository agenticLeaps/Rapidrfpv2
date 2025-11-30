#!/usr/bin/env python3
"""
NeonDB Storage for NodeRAG - handles graph and embedding storage in PostgreSQL
"""

import os
import asyncio
import asyncpg
import pickle
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from ..config.settings import Config
from ..graph.node_types import NodeType
from ..llm.llm_service import LLMService

logger = logging.getLogger(__name__)

class NeonStorage:
    def __init__(self):
        self.db_url = os.getenv("NEON_DATABASE_URL")
        if not self.db_url:
            raise ValueError("NEON_DATABASE_URL environment variable not found")
        
        self.llm_service = LLMService()
        
    async def _get_connection(self):
        """Get database connection"""
        return await asyncpg.connect(self.db_url)
    
    async def _ensure_tables_exist(self):
        """Ensure NodeRAG tables exist in database"""
        conn = await self._get_connection()
        try:
            # Create noderag_embeddings table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS noderag_embeddings (
                    id SERIAL PRIMARY KEY,
                    node_id VARCHAR(255) UNIQUE NOT NULL,
                    node_type VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    org_id VARCHAR(255) NOT NULL,
                    file_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    chunk_index INTEGER,
                    graph_metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create noderag_graphs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS noderag_graphs (
                    id SERIAL PRIMARY KEY,
                    file_id VARCHAR(255) NOT NULL,
                    org_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    graph_data BYTEA,
                    stats JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(org_id, file_id)
                );
            """)
            
            # Add updated_at column and token tracking columns to existing noderag_graphs table if they don't exist
            try:
                await conn.execute("""
                    ALTER TABLE noderag_graphs 
                    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ADD COLUMN IF NOT EXISTS input_tokens INTEGER DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS output_tokens INTEGER DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS total_tokens INTEGER DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS api_calls INTEGER DEFAULT 0
                """)
            except Exception as e:
                logger.warning(f"Could not add columns (might already exist): {e}")
            
            # Create indexes for better performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_noderag_embeddings_org_file ON noderag_embeddings(org_id, file_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_noderag_embeddings_node_type ON noderag_embeddings(node_type);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_noderag_graphs_org_file ON noderag_graphs(org_id, file_id);")
            
            logger.info("✅ NodeRAG database tables ensured")
            
        finally:
            await conn.close()
    
    def store_noderag_data(self, org_id: str, file_id: str, user_id: str, pipeline, 
                          input_tokens: int = 0, output_tokens: int = 0, api_calls: int = 0) -> Dict[str, Any]:
        """Store NodeRAG graph and embeddings data with token tracking"""
        return asyncio.run(self._store_noderag_data_async(org_id, file_id, user_id, pipeline, 
                                                         input_tokens, output_tokens, api_calls))
    
    async def _store_noderag_data_async(self, org_id: str, file_id: str, user_id: str, pipeline, 
                                       input_tokens: int = 0, output_tokens: int = 0, api_calls: int = 0) -> Dict[str, Any]:
        """Async implementation of store_noderag_data with token tracking"""
        try:
            await self._ensure_tables_exist()
            conn = await self._get_connection()
            
            stored_embeddings = 0
            stored_graphs = 0
            
            try:
                # Start transaction
                async with conn.transaction():
                    
                    # 1. Store graph data with token tracking
                    graph_data = pickle.dumps(pipeline.graph_manager.graph)
                    graph_stats = pipeline.graph_manager.get_stats()
                    total_tokens = input_tokens + output_tokens
                    
                    await conn.execute("""
                        INSERT INTO noderag_graphs 
                        (file_id, org_id, user_id, graph_data, stats, input_tokens, output_tokens, total_tokens, api_calls)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (org_id, file_id) 
                        DO UPDATE SET 
                            graph_data = EXCLUDED.graph_data,
                            stats = EXCLUDED.stats,
                            input_tokens = EXCLUDED.input_tokens,
                            output_tokens = EXCLUDED.output_tokens,
                            total_tokens = EXCLUDED.total_tokens,
                            api_calls = EXCLUDED.api_calls,
                            updated_at = CURRENT_TIMESTAMP
                    """, file_id, org_id, user_id, graph_data, json.dumps(graph_stats), 
                        input_tokens, output_tokens, total_tokens, api_calls)
                    
                    stored_graphs = 1
                    logger.info(f"✅ Stored graph data for file_id={file_id}")
                    
                    # 2. Store node embeddings
                    all_nodes = []
                    for node_type in [NodeType.SEMANTIC, NodeType.ENTITY, NodeType.RELATIONSHIP, 
                                     NodeType.ATTRIBUTE, NodeType.HIGH_LEVEL, NodeType.OVERVIEW]:
                        try:
                            nodes = pipeline.graph_manager.get_nodes_by_type(node_type)
                            all_nodes.extend(nodes)
                        except Exception as e:
                            logger.warning(f"Could not get nodes of type {node_type}: {e}")
                    
                    logger.info(f"📊 Found {len(all_nodes)} nodes to store")
                    
                    # Clear existing embeddings for this file
                    await conn.execute(
                        "DELETE FROM noderag_embeddings WHERE org_id = $1 AND file_id = $2",
                        org_id, file_id
                    )
                    
                    # Store new embeddings in optimized batch operations
                    embedding_records = []
                    for node in all_nodes:
                        if hasattr(node, 'embeddings') and node.embeddings is not None:
                            try:
                                # Convert embedding to string format for vector type
                                embedding_str = '[' + ','.join(map(str, node.embeddings)) + ']'
                                
                                # Prepare graph metadata
                                graph_metadata = {
                                    'node_metadata': node.metadata if hasattr(node, 'metadata') else {},
                                    'node_type_value': node.type.value if hasattr(node.type, 'value') else str(node.type)
                                }
                                
                                embedding_records.append((
                                    node.id,
                                    node.type.value if hasattr(node.type, 'value') else str(node.type),
                                    node.content,
                                    embedding_str,
                                    org_id,
                                    file_id,
                                    user_id,
                                    getattr(node, 'chunk_index', 0),
                                    json.dumps(graph_metadata)
                                ))
                                
                            except Exception as e:
                                logger.warning(f"Failed to prepare embedding for node {node.id}: {e}")
                    
                    # Batch insert all embeddings for better performance
                    if embedding_records:
                        await conn.executemany("""
                            INSERT INTO noderag_embeddings 
                            (node_id, node_type, content, embedding, org_id, file_id, user_id, 
                             chunk_index, graph_metadata)
                            VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9)
                            ON CONFLICT (node_id) DO UPDATE SET
                                node_type = EXCLUDED.node_type,
                                content = EXCLUDED.content,
                                embedding = EXCLUDED.embedding,
                                org_id = EXCLUDED.org_id,
                                file_id = EXCLUDED.file_id,
                                user_id = EXCLUDED.user_id,
                                chunk_index = EXCLUDED.chunk_index,
                                graph_metadata = EXCLUDED.graph_metadata,
                                updated_at = CURRENT_TIMESTAMP
                        """, embedding_records)
                        stored_embeddings = len(embedding_records)
                    
                    logger.info(f"✅ Stored {stored_embeddings} embeddings for file_id={file_id}")
                
            finally:
                await conn.close()
            
            return {
                "success": True,
                "embeddings_stored": stored_embeddings,
                "graphs_stored": stored_graphs,
                "graph_nodes": len(all_nodes)
            }
            
        except Exception as e:
            logger.error(f"❌ Error storing NodeRAG data: {e}")
            return {
                "success": False,
                "error": str(e),
                "embeddings_stored": stored_embeddings,
                "graphs_stored": stored_graphs
            }
    
    def search_noderag_data(self, org_id: str, query: str, top_k: int = 10, filters: Dict = None) -> List[Dict]:
        """Search NodeRAG embeddings"""
        return asyncio.run(self._search_noderag_data_async(org_id, query, top_k, filters))
    
    async def _search_noderag_data_async(self, org_id: str, query: str, top_k: int = 10, filters: Dict = None) -> List[Dict]:
        """Async implementation of search_noderag_data"""
        try:
            # Generate query embedding
            query_embeddings = self.llm_service.get_embeddings([query])
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            conn = await self._get_connection()
            
            try:
                # Build query with filters
                where_clause = "WHERE org_id = $1"
                params = [org_id]
                param_count = 1
                
                if filters:
                    if 'file_id' in filters:
                        param_count += 1
                        where_clause += f" AND file_id = ${param_count}"
                        params.append(filters['file_id'])
                    
                    if 'node_type' in filters:
                        param_count += 1
                        where_clause += f" AND node_type = ${param_count}"
                        params.append(filters['node_type'])
                
                param_count += 1
                query_sql = f"""
                    SELECT 
                        node_id,
                        node_type,
                        content,
                        file_id,
                        user_id,
                        chunk_index,
                        graph_metadata,
                        (embedding <-> ${param_count}::vector) as distance
                    FROM noderag_embeddings 
                    {where_clause}
                    ORDER BY embedding <-> ${param_count}::vector
                    LIMIT {top_k}
                """
                
                params.append(query_embedding_str)
                
                rows = await conn.fetch(query_sql, *params)
                
                results = []
                for row in rows:
                    results.append({
                        "node_id": row['node_id'],
                        "node_type": row['node_type'],
                        "content": row['content'],
                        "file_id": row['file_id'],
                        "user_id": row['user_id'],
                        "chunk_index": row['chunk_index'],
                        "metadata": json.loads(row['graph_metadata']) if row['graph_metadata'] else {},
                        "similarity_score": 1.0 - float(row['distance'])  # Convert distance to similarity
                    })
                
                logger.info(f"🔍 Found {len(results)} search results for query: '{query[:50]}...'")
                return results
                
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    def delete_file_data(self, org_id: str, file_id: str) -> Dict[str, Any]:
        """Delete all data for a specific file"""
        return asyncio.run(self._delete_file_data_async(org_id, file_id))
    
    async def _delete_file_data_async(self, org_id: str, file_id: str) -> Dict[str, Any]:
        """Async implementation of delete_file_data"""
        try:
            conn = await self._get_connection()
            
            try:
                async with conn.transaction():
                    # Delete embeddings
                    embeddings_result = await conn.execute(
                        "DELETE FROM noderag_embeddings WHERE org_id = $1 AND file_id = $2",
                        org_id, file_id
                    )
                    
                    # Delete graph data
                    graph_result = await conn.execute(
                        "DELETE FROM noderag_graphs WHERE org_id = $1 AND file_id = $2",
                        org_id, file_id
                    )
                    
                    # Extract deleted count from result
                    embeddings_deleted = int(embeddings_result.split()[-1]) if embeddings_result.split()[-1].isdigit() else 0
                    graphs_deleted = int(graph_result.split()[-1]) if graph_result.split()[-1].isdigit() else 0
                    
                    logger.info(f"🗑️ Deleted {embeddings_deleted} embeddings and {graphs_deleted} graphs for file_id={file_id}")
                    
                    return {
                        "success": True,
                        "deleted_count": embeddings_deleted,
                        "graphs_deleted": graphs_deleted
                    }
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_count": 0
            }
    
    def inspect_all_data(self) -> Dict[str, Any]:
        """Inspect all data in the database for debugging"""
        return asyncio.run(self._inspect_all_data_async())
    
    async def _inspect_all_data_async(self) -> Dict[str, Any]:
        """Async implementation of inspect all data"""
        try:
            conn = await self._get_connection()
            
            try:
                # Get summary of all data
                summary_query = """
                SELECT file_id, org_id, node_type, COUNT(*) as count,
                       MIN(created_at) as first_created, MAX(created_at) as last_created
                FROM noderag_embeddings 
                GROUP BY file_id, org_id, node_type
                ORDER BY last_created DESC
                LIMIT 50
                """
                
                rows = await conn.fetch(summary_query)
                
                results = []
                for row in rows:
                    results.append({
                        "file_id": row['file_id'],
                        "org_id": row['org_id'], 
                        "node_type": row['node_type'],
                        "count": row['count'],
                        "first_created": str(row['first_created']),
                        "last_created": str(row['last_created'])
                    })
                
                return {
                    "success": True,
                    "results": results,
                    "total_groups": len(results)
                }
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"❌ Inspect error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_file_stats(self, org_id: str, file_id: str = None) -> Dict[str, Any]:
        """Get statistics for files"""
        return asyncio.run(self._get_file_stats_async(org_id, file_id))
    
    async def _get_file_stats_async(self, org_id: str, file_id: str = None) -> Dict[str, Any]:
        """Async implementation of get_file_stats"""
        try:
            conn = await self._get_connection()
            
            try:
                if file_id:
                    # Get stats for specific file
                    embedding_count = await conn.fetchval(
                        "SELECT COUNT(*) FROM noderag_embeddings WHERE org_id = $1 AND file_id = $2",
                        org_id, file_id
                    )
                    
                    graph_info = await conn.fetchrow(
                        "SELECT stats FROM noderag_graphs WHERE org_id = $1 AND file_id = $2",
                        org_id, file_id
                    )
                    
                    return {
                        "org_id": org_id,
                        "file_id": file_id,
                        "embedding_count": embedding_count,
                        "graph_stats": json.loads(graph_info['stats']) if graph_info and graph_info['stats'] else {}
                    }
                else:
                    # Get stats for all files in org
                    total_embeddings = await conn.fetchval(
                        "SELECT COUNT(*) FROM noderag_embeddings WHERE org_id = $1",
                        org_id
                    )
                    
                    total_graphs = await conn.fetchval(
                        "SELECT COUNT(*) FROM noderag_graphs WHERE org_id = $1",
                        org_id
                    )
                    
                    files = await conn.fetch(
                        "SELECT file_id, COUNT(*) as embedding_count FROM noderag_embeddings WHERE org_id = $1 GROUP BY file_id",
                        org_id
                    )
                    
                    return {
                        "org_id": org_id,
                        "total_embeddings": total_embeddings,
                        "total_graphs": total_graphs,
                        "files": [{"file_id": row['file_id'], "embedding_count": row['embedding_count']} for row in files]
                    }
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {"error": str(e)}
    
    def store_api_usage(self, org_id: str, user_id: str, endpoint: str, 
                       input_tokens: int, output_tokens: int, total_tokens: int, 
                       api_calls: int, metadata: dict = None) -> Dict[str, Any]:
        """Store API usage tracking data"""
        return asyncio.run(self._store_api_usage_async(
            org_id, user_id, endpoint, input_tokens, output_tokens, 
            total_tokens, api_calls, metadata
        ))
    
    async def _store_api_usage_async(self, org_id: str, user_id: str, endpoint: str, 
                                    input_tokens: int, output_tokens: int, total_tokens: int, 
                                    api_calls: int, metadata: dict = None) -> Dict[str, Any]:
        """Async implementation of store_api_usage"""
        try:
            await self._ensure_api_usage_table_exists()
            conn = await self._get_connection()
            
            try:
                await conn.execute("""
                    INSERT INTO api_usage_tracking 
                    (org_id, user_id, endpoint, input_tokens, output_tokens, total_tokens, api_calls, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, org_id, user_id, endpoint, input_tokens, output_tokens, 
                    total_tokens, api_calls, json.dumps(metadata) if metadata else None)
                
                logger.info(f"✅ Stored API usage: {endpoint} - tokens={total_tokens}, calls={api_calls}")
                return {"success": True}
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"❌ API usage storage error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _ensure_api_usage_table_exists(self):
        """Ensure API usage tracking table exists"""
        try:
            conn = await self._get_connection()
            
            try:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS api_usage_tracking (
                        id SERIAL PRIMARY KEY,
                        org_id VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        endpoint VARCHAR(100) NOT NULL,
                        input_tokens INTEGER DEFAULT 0,
                        output_tokens INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        api_calls INTEGER DEFAULT 0,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index for better performance
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_org_endpoint ON api_usage_tracking(org_id, endpoint);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_created ON api_usage_tracking(created_at);")
                
                logger.debug("✅ API usage tracking table ensured")
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.warning(f"Could not create API usage table: {e}")
    
    def get_storage_stats(self, org_id: str) -> Dict[str, Any]:
        """Get storage statistics for a specific organization"""
        stats_result = self.get_stats(org_id)
        return {
            "success": True,
            "stats": stats_result
        }