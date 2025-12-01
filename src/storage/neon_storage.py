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
import time

from ..config.settings import Config
from ..graph.node_types import NodeType
from ..llm.llm_service import LLMService

logger = logging.getLogger(__name__)

class NeonDBStorage:
    def __init__(self):
        self.db_url = os.getenv("NEON_DATABASE_URL")
        if not self.db_url:
            raise ValueError("NEON_DATABASE_URL environment variable not found")
        
        self.llm_service = LLMService()
        self._connection_pool = None
        
    async def _get_connection_pool(self):
        """Get or create database connection pool"""
        if self._connection_pool is None:
            self._connection_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                max_inactive_connection_lifetime=300.0,
                command_timeout=60
            )
        return self._connection_pool
        
    async def _get_connection(self):
        """Get database connection from pool"""
        pool = await self._get_connection_pool()
        return await pool.acquire()
        
    async def _release_connection(self, conn):
        """Release connection back to pool"""
        if self._connection_pool:
            await self._connection_pool.release(conn)
    
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
            
            # Add updated_at column to existing noderag_graphs table if it doesn't exist
            try:
                await conn.execute("""
                    ALTER TABLE noderag_graphs 
                    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """)
            except Exception as e:
                logger.warning(f"Could not add updated_at column (might already exist): {e}")
            
            # Create optimized indexes for better performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_noderag_embeddings_org_file ON noderag_embeddings(org_id, file_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_noderag_embeddings_node_type ON noderag_embeddings(node_type);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_noderag_embeddings_org_id ON noderag_embeddings(org_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_noderag_graphs_org_file ON noderag_graphs(org_id, file_id);")
            
            # Add HNSW index for vector similarity search if extension is available
            try:
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_noderag_embeddings_vector_hnsw ON noderag_embeddings USING hnsw (embedding vector_cosine_ops);")
                logger.info("✅ Created HNSW vector index for faster similarity search")
            except Exception as e:
                logger.warning(f"HNSW extension not available: {e}")
            
            logger.info("✅ NodeRAG database tables ensured")
            
        finally:
            await self._release_connection(conn)
    
    def store_noderag_data(self, org_id: str, file_id: str, user_id: str, pipeline) -> Dict[str, Any]:
        """Store NodeRAG graph and embeddings data"""
        return asyncio.run(self._store_noderag_data_async(org_id, file_id, user_id, pipeline))
    
    async def _store_noderag_data_async(self, org_id: str, file_id: str, user_id: str, pipeline) -> Dict[str, Any]:
        """Async implementation of store_noderag_data with optimized batch operations"""
        start_time = time.time()
        stored_embeddings = 0
        stored_graphs = 0
        all_nodes = []
        
        try:
            await self._ensure_tables_exist()
            conn = await self._get_connection()
            
            try:
                # Start transaction
                async with conn.transaction():
                    
                    # 1. Store graph data
                    graph_data = pickle.dumps(pipeline.graph_manager.graph)
                    graph_stats = pipeline.graph_manager.get_stats()
                    
                    await conn.execute("""
                        INSERT INTO noderag_graphs 
                        (file_id, org_id, user_id, graph_data, stats)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (org_id, file_id) 
                        DO UPDATE SET 
                            graph_data = EXCLUDED.graph_data,
                            stats = EXCLUDED.stats,
                            updated_at = CURRENT_TIMESTAMP
                    """, file_id, org_id, user_id, graph_data, json.dumps(graph_stats))
                    
                    stored_graphs = 1
                    logger.info(f"✅ Stored graph data for file_id={file_id}")
                    
                    # 2. Collect all nodes first
                    for node_type in [NodeType.SEMANTIC, NodeType.ENTITY, NodeType.RELATIONSHIP, 
                                     NodeType.ATTRIBUTE, NodeType.HIGH_LEVEL, NodeType.OVERVIEW]:
                        try:
                            nodes = pipeline.graph_manager.get_nodes_by_type(node_type)
                            all_nodes.extend(nodes)
                        except Exception as e:
                            logger.warning(f"Could not get nodes of type {node_type}: {e}")
                    
                    logger.info(f"📊 Found {len(all_nodes)} nodes to store")
                    
                    # 3. Clear existing embeddings for this file (single operation)
                    delete_start = time.time()
                    await conn.execute(
                        "DELETE FROM noderag_embeddings WHERE org_id = $1 AND file_id = $2",
                        org_id, file_id
                    )
                    logger.info(f"🗑️ Cleared existing embeddings in {time.time() - delete_start:.2f}s")
                    
                    # 4. Prepare batch data for embeddings
                    batch_data = []
                    for i, node in enumerate(all_nodes):
                        if hasattr(node, 'embeddings') and node.embeddings is not None:
                            try:
                                # Convert embedding to string format for vector type
                                embedding_str = '[' + ','.join(map(str, node.embeddings)) + ']'
                                
                                # Prepare graph metadata
                                graph_metadata = {
                                    'node_metadata': node.metadata if hasattr(node, 'metadata') else {},
                                    'node_type_value': node.type.value if hasattr(node.type, 'value') else str(node.type)
                                }
                                
                                batch_data.append((
                                    node.id,
                                    node.type.value if hasattr(node.type, 'value') else str(node.type),
                                    node.content[:2000],  # Limit content length
                                    embedding_str,
                                    org_id,
                                    file_id,
                                    user_id,
                                    i,  # chunk_index
                                    json.dumps(graph_metadata)
                                ))
                                
                            except Exception as e:
                                logger.error(f"Error preparing embedding for node {node.id}: {e}")
                                continue
                    
                    # 5. Batch insert embeddings (much faster than individual inserts)
                    if batch_data:
                        batch_start = time.time()
                        
                        # Use executemany for batch insert with conflict resolution
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
                        """, batch_data)
                        
                        stored_embeddings = len(batch_data)
                        batch_time = time.time() - batch_start
                        logger.info(f"✅ Batch inserted {stored_embeddings} embeddings in {batch_time:.2f}s ({stored_embeddings/batch_time:.1f} ops/sec)")
                    
                    total_time = time.time() - start_time
                    logger.info(f"✅ Total storage time: {total_time:.2f}s for file_id={file_id}")
                
            finally:
                await self._release_connection(conn)
            
            return {
                "success": True,
                "embeddings_stored": stored_embeddings,
                "graphs_stored": stored_graphs,
                "graph_nodes": len(all_nodes),
                "storage_time_seconds": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error storing NodeRAG data: {e}")
            return {
                "success": False,
                "error": str(e),
                "embeddings_stored": stored_embeddings,
                "graphs_stored": stored_graphs,
                "storage_time_seconds": time.time() - start_time
            }
    
    def search_noderag_data(self, org_id: str, query: str, top_k: int = 10, filters: Dict = None) -> List[Dict]:
        """Search NodeRAG embeddings"""
        return asyncio.run(self._search_noderag_data_async(org_id, query, top_k, filters))
    
    async def _search_noderag_data_async(self, org_id: str, query: str, top_k: int = 10, filters: Dict = None) -> List[Dict]:
        """Async implementation of search_noderag_data with optimized connection pooling"""
        search_start = time.time()
        
        try:
            # Generate query embedding
            query_embeddings = self.llm_service.get_embeddings([query])
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            conn = await self._get_connection()
            
            try:
                # Build optimized query with filters
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
                
                # Use optimized query with HNSW index if available
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
                
                search_time = time.time() - search_start
                logger.info(f"🔍 Found {len(results)} search results in {search_time:.2f}s for query: '{query[:50]}...'")
                return results
                
            finally:
                await self._release_connection(conn)
        
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    def delete_file_data(self, org_id: str, file_id: str) -> Dict[str, Any]:
        """Delete all data for a specific file"""
        return asyncio.run(self._delete_file_data_async(org_id, file_id))
    
    async def _delete_file_data_async(self, org_id: str, file_id: str) -> Dict[str, Any]:
        """Async implementation of delete_file_data with optimized connection pooling"""
        delete_start = time.time()
        
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
                    
                    delete_time = time.time() - delete_start
                    logger.info(f"🗑️ Deleted {embeddings_deleted} embeddings and {graphs_deleted} graphs for file_id={file_id} in {delete_time:.2f}s")
                    
                    return {
                        "success": True,
                        "deleted_count": embeddings_deleted,
                        "graphs_deleted": graphs_deleted,
                        "delete_time_seconds": delete_time
                    }
            
            finally:
                await self._release_connection(conn)
        
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_count": 0,
                "delete_time_seconds": time.time() - delete_start
            }
    
    def inspect_all_data(self) -> Dict[str, Any]:
        """Inspect all data in the database for debugging"""
        return asyncio.run(self._inspect_all_data_async())
    
    async def _inspect_all_data_async(self) -> Dict[str, Any]:
        """Async implementation of inspect all data with connection pooling"""
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
                await self._release_connection(conn)
        
        except Exception as e:
            logger.error(f"❌ Inspect error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_file_stats(self, org_id: str, file_id: str = None) -> Dict[str, Any]:
        """Get statistics for files"""
        return asyncio.run(self._get_file_stats_async(org_id, file_id))
    
    async def _get_file_stats_async(self, org_id: str, file_id: str = None) -> Dict[str, Any]:
        """Async implementation of get_file_stats with connection pooling"""
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
                await self._release_connection(conn)
        
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {"error": str(e)}
    
    async def close_pool(self):
        """Close connection pool"""
        if self._connection_pool:
            await self._connection_pool.close()
            self._connection_pool = None
            logger.info("✅ Database connection pool closed")
    
    async def bulk_store_embeddings_async(self, org_id: str, file_id: str, user_id: str, embedding_data: List[Dict]) -> Dict[str, Any]:
        """Optimized bulk storage method for large datasets"""
        start_time = time.time()
        stored_embeddings = 0
        
        try:
            conn = await self._get_connection()
            
            try:
                async with conn.transaction():
                    # Clear existing embeddings for this file
                    delete_start = time.time()
                    await conn.execute(
                        "DELETE FROM noderag_embeddings WHERE org_id = $1 AND file_id = $2",
                        org_id, file_id
                    )
                    logger.info(f"🗑️ Cleared existing embeddings in {time.time() - delete_start:.2f}s")
                    
                    if not embedding_data:
                        return {"success": True, "embeddings_stored": 0, "storage_time_seconds": time.time() - start_time}
                    
                    # Prepare batch data with chunking for memory efficiency
                    batch_size = 1000  # Process in chunks of 1000
                    total_batches = (len(embedding_data) + batch_size - 1) // batch_size
                    
                    for batch_idx in range(0, len(embedding_data), batch_size):
                        batch_start_time = time.time()
                        batch_chunk = embedding_data[batch_idx:batch_idx + batch_size]
                        
                        batch_data = []
                        for i, item in enumerate(batch_chunk):
                            try:
                                # Prepare embedding string
                                embedding_str = '[' + ','.join(map(str, item['embedding'])) + ']'
                                
                                batch_data.append((
                                    item['node_id'],
                                    item['node_type'],
                                    item['content'][:2000],  # Limit content length
                                    embedding_str,
                                    org_id,
                                    file_id,
                                    user_id,
                                    batch_idx + i,  # chunk_index
                                    json.dumps(item.get('metadata', {}))
                                ))
                                
                            except Exception as e:
                                logger.error(f"Error preparing embedding for batch item {i}: {e}")
                                continue
                        
                        if batch_data:
                            # Batch insert with conflict resolution
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
                            """, batch_data)
                            
                            stored_embeddings += len(batch_data)
                            batch_time = time.time() - batch_start_time
                            logger.info(f"✅ Batch {(batch_idx//batch_size)+1}/{total_batches}: {len(batch_data)} embeddings in {batch_time:.2f}s")
                    
                    total_time = time.time() - start_time
                    throughput = stored_embeddings / total_time if total_time > 0 else 0
                    logger.info(f"✅ Bulk stored {stored_embeddings} embeddings in {total_time:.2f}s ({throughput:.1f} ops/sec)")
                
            finally:
                await self._release_connection(conn)
            
            return {
                "success": True,
                "embeddings_stored": stored_embeddings,
                "storage_time_seconds": time.time() - start_time,
                "throughput_ops_per_sec": throughput
            }
            
        except Exception as e:
            logger.error(f"❌ Error in bulk storage: {e}")
            return {
                "success": False,
                "error": str(e),
                "embeddings_stored": stored_embeddings,
                "storage_time_seconds": time.time() - start_time
            }
    
    def bulk_store_embeddings(self, org_id: str, file_id: str, user_id: str, embedding_data: List[Dict]) -> Dict[str, Any]:
        """Sync wrapper for bulk storage"""
        return asyncio.run(self.bulk_store_embeddings_async(org_id, file_id, user_id, embedding_data))