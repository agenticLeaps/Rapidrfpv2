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
import threading

# Add sync database support for org operations
try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from ..config.settings import Config
from ..graph.node_types import NodeType
from ..llm.llm_service import LLMService

logger = logging.getLogger(__name__)

def run_async_safe(coro):
    """Safely run async function in sync context using a dedicated thread"""
    import concurrent.futures
    import threading
    
    def run_in_dedicated_thread():
        # Create a completely new event loop in a dedicated thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    # Always use a separate thread to avoid any event loop conflicts
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_dedicated_thread)
        return future.result(timeout=120)  # 2 minute timeout

class NeonDBStorage:
    def __init__(self):
        self.db_url = os.getenv("NEON_DATABASE_URL")
        if not self.db_url:
            raise ValueError("NEON_DATABASE_URL environment variable not found")
        
        self.llm_service = LLMService()
        self._connection_pool = None
        
    async def _get_connection_pool(self):
        """Get or create database connection pool"""
        if self._connection_pool is None or self._connection_pool.is_closing():
            # Close existing pool if it's in a bad state
            if self._connection_pool and not self._connection_pool.is_closing():
                try:
                    await self._connection_pool.close()
                except Exception:
                    pass
            
            self._connection_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=1,
                max_size=5,  # Reduced to avoid connection conflicts
                max_inactive_connection_lifetime=120.0,  # Shorter timeout
                command_timeout=120,  # Increased for large operations
                max_queries=10000,
                max_cached_statement_lifetime=300
            )
            logger.info("✅ Database connection pool created")
        return self._connection_pool
        
    async def _get_connection(self):
        """Get database connection from pool with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pool = await self._get_connection_pool()
                conn = await pool.acquire()
                return conn
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    # Last attempt - try direct connection
                    logger.info("Falling back to direct connection")
                    return await asyncpg.connect(self.db_url)
                await asyncio.sleep(1)
        
    async def _release_connection(self, conn):
        """Release connection back to pool"""
        try:
            if self._connection_pool and not self._connection_pool.is_closing():
                await self._connection_pool.release(conn)
            else:
                await conn.close()
        except Exception as e:
            logger.warning(f"Error releasing connection: {e}")
            try:
                await conn.close()
            except Exception:
                pass
    
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
            
            # Create noderag_graphs table for org-level storage
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS noderag_graphs (
                    id SERIAL PRIMARY KEY,
                    org_id VARCHAR(255) NOT NULL UNIQUE,
                    user_id VARCHAR(255) NOT NULL,
                    graph_data BYTEA,
                    stats JSONB,
                    processed_files JSONB DEFAULT '[]'::jsonb,
                    version INTEGER DEFAULT 1,
                    last_file_added VARCHAR(255),
                    last_incremental_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Migrate existing noderag_graphs table if needed
            await self._migrate_existing_graphs_table(conn)
            
            # Create file processing tracking table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS noderag_file_processing (
                    id SERIAL PRIMARY KEY,
                    org_id VARCHAR(255) NOT NULL,
                    file_id VARCHAR(255) NOT NULL,
                    file_hash VARCHAR(64),
                    processing_status VARCHAR(20) DEFAULT 'pending',
                    node_count INTEGER DEFAULT 0,
                    processing_time FLOAT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    UNIQUE(org_id, file_id)
                );
            """)
            
            logger.info("✅ NodeRAG tables ensured")
            
        finally:
            await self._release_connection(conn)
    
    async def _migrate_existing_graphs_table(self, conn):
        """Migrate existing noderag_graphs table to support org-level storage"""
        try:
            # Check if processed_files column exists
            result = await conn.fetchval("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'noderag_graphs' AND column_name = 'processed_files'
            """)
            
            if not result:
                logger.info("🔄 Migrating noderag_graphs table for org-level support...")
                
                # Add new columns
                await conn.execute("""
                    ALTER TABLE noderag_graphs 
                    ADD COLUMN IF NOT EXISTS processed_files JSONB DEFAULT '[]'::jsonb,
                    ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1,
                    ADD COLUMN IF NOT EXISTS last_file_added VARCHAR(255),
                    ADD COLUMN IF NOT EXISTS last_incremental_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """)
                
                # Remove old file_id constraint and add org_id constraint
                await conn.execute("""
                    ALTER TABLE noderag_graphs DROP CONSTRAINT IF EXISTS noderag_graphs_org_id_file_id_key
                """)
                
                # Update existing records to have file_id in processed_files
                await conn.execute("""
                    UPDATE noderag_graphs 
                    SET processed_files = jsonb_build_array(
                        COALESCE(
                            (SELECT file_id FROM noderag_graphs ng2 WHERE ng2.org_id = noderag_graphs.org_id LIMIT 1),
                            'unknown_file'
                        )
                    )
                    WHERE processed_files IS NULL OR processed_files = '[]'::jsonb
                """)
                
                logger.info("✅ noderag_graphs table migration completed")
            
        except Exception as e:
            logger.warning(f"Migration warning: {e}")
            # Continue anyway - table might already be in correct format
    
    def run_migration(self):
        """Public method to run migration manually if needed"""
        return asyncio.run(self._run_migration_async())
    
    async def _run_migration_async(self):
        """Run migration in async context"""
        conn = await self._get_connection()
        try:
            await self._migrate_existing_graphs_table(conn)
            return True
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
        finally:
            await self._release_connection(conn)
    
    def store_noderag_data(self, org_id: str, file_id: str, user_id: str, pipeline) -> Dict[str, Any]:
        """Store NodeRAG graph and embeddings data"""
        return asyncio.run(self._store_noderag_data_async(org_id, file_id, user_id, pipeline))
    
    async def _store_noderag_data_async(self, org_id: str, file_id: str, user_id: str, pipeline) -> Dict[str, Any]:
        """Async implementation of store_noderag_data with optimized batch operations and improved error handling"""
        start_time = time.time()
        stored_embeddings = 0
        stored_graphs = 0
        all_nodes = []
        conn = None
        
        try:
            await self._ensure_tables_exist()
            
            # Get connection with retry logic
            conn = await self._get_connection()
            logger.info(f"📡 Got database connection for storage")
            
            # Use a more robust transaction approach
            async with conn.transaction(isolation='read_committed'):
                logger.info(f"🔒 Started transaction for file_id={file_id}")
                
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
                
                # 4. Process embeddings in smaller chunks to avoid memory issues
                chunk_size = 500  # Process in chunks to avoid overwhelming the connection
                total_chunks = (len(all_nodes) + chunk_size - 1) // chunk_size
                
                for chunk_idx in range(0, len(all_nodes), chunk_size):
                    chunk_start = chunk_idx
                    chunk_end = min(chunk_idx + chunk_size, len(all_nodes))
                    chunk_nodes = all_nodes[chunk_start:chunk_end]
                    
                    # Prepare batch data for this chunk
                    batch_data = []
                    for i, node in enumerate(chunk_nodes):
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
                                    chunk_start + i,  # chunk_index
                                    json.dumps(graph_metadata)
                                ))
                                
                            except Exception as e:
                                logger.error(f"Error preparing embedding for node {node.id}: {e}")
                                continue
                    
                    # Insert this chunk
                    if batch_data:
                        chunk_batch_start = time.time()
                        
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
                        chunk_time = time.time() - chunk_batch_start
                        chunk_num = (chunk_idx // chunk_size) + 1
                        logger.info(f"✅ DB chunk {chunk_num}/{total_chunks}: {len(batch_data)} embeddings in {chunk_time:.2f}s")
                
                total_time = time.time() - start_time
                logger.info(f"✅ Transaction completed: {stored_embeddings} embeddings stored in {total_time:.2f}s")
            
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
        finally:
            if conn:
                try:
                    await self._release_connection(conn)
                except Exception as e:
                    logger.warning(f"Error releasing connection: {e}")
    
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

    def delete_org_data(self, org_id: str) -> Dict[str, Any]:
        """Delete all data for an entire organization"""
        return asyncio.run(self._delete_org_data_async(org_id))

    async def _delete_org_data_async(self, org_id: str) -> Dict[str, Any]:
        """Async implementation of delete_org_data - deletes all org data"""
        delete_start = time.time()

        try:
            conn = await self._get_connection()

            try:
                async with conn.transaction():
                    # 1. Delete all embeddings for this org
                    embeddings_result = await conn.execute(
                        "DELETE FROM noderag_embeddings WHERE org_id = $1",
                        org_id
                    )

                    embeddings_deleted = int(embeddings_result.split()[-1]) if embeddings_result.split()[-1].isdigit() else 0

                    # 2. Delete org graph
                    graph_result = await conn.execute(
                        "DELETE FROM noderag_graphs WHERE org_id = $1",
                        org_id
                    )

                    graph_deleted = int(graph_result.split()[-1]) if graph_result.split()[-1].isdigit() else 0

                    # 3. Delete file processing records
                    processing_result = await conn.execute(
                        "DELETE FROM noderag_file_processing WHERE org_id = $1",
                        org_id
                    )

                    processing_deleted = int(processing_result.split()[-1]) if processing_result.split()[-1].isdigit() else 0

                    delete_time = time.time() - delete_start
                    logger.info(f"🗑️ Deleted entire org {org_id}: {embeddings_deleted} embeddings, {graph_deleted} graphs, {processing_deleted} processing records in {delete_time:.2f}s")

                    return {
                        "success": True,
                        "embeddings_deleted": embeddings_deleted,
                        "graphs_deleted": graph_deleted,
                        "processing_records_deleted": processing_deleted,
                        "delete_time_seconds": delete_time
                    }

            finally:
                await self._release_connection(conn)

        except Exception as e:
            logger.error(f"❌ Delete org error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "embeddings_deleted": 0,
                "delete_time_seconds": time.time() - delete_start
            }
    
    async def _delete_file_data_async(self, org_id: str, file_id: str) -> Dict[str, Any]:
        """Async implementation of delete_file_data with org-level graph handling"""
        delete_start = time.time()

        try:
            conn = await self._get_connection()

            try:
                async with conn.transaction():
                    # 1. Delete embeddings for this file
                    embeddings_result = await conn.execute(
                        "DELETE FROM noderag_embeddings WHERE org_id = $1 AND file_id = $2",
                        org_id, file_id
                    )

                    embeddings_deleted = int(embeddings_result.split()[-1]) if embeddings_result.split()[-1].isdigit() else 0

                    # 2. Load org graph to update it
                    org_graph_row = await conn.fetchrow(
                        "SELECT graph_data, processed_files, version, user_id, stats FROM noderag_graphs WHERE org_id = $1",
                        org_id
                    )

                    nodes_removed = 0
                    graph_updated = False

                    if org_graph_row:
                        try:
                            # Deserialize graph data
                            graph_data_dict = pickle.loads(org_graph_row['graph_data'])
                            processed_files = org_graph_row['processed_files'] if org_graph_row['processed_files'] else []

                            # Check if file is in processed list
                            if file_id in processed_files:
                                # Load graph into GraphManager to remove file nodes
                                from ..graph.graph_manager import GraphManager
                                graph_manager = GraphManager()
                                graph_manager.graph = graph_data_dict['graph']
                                graph_manager.entity_index = graph_data_dict.get('entity_index', {})
                                graph_manager.community_assignments = graph_data_dict.get('community_assignments', {})

                                # Remove nodes for this file
                                clear_result = graph_manager.clear_file_nodes(file_id)
                                nodes_removed = clear_result.get('nodes_removed', 0)

                                # Update processed files list
                                processed_files.remove(file_id)

                                # Check if any files remain
                                if len(processed_files) == 0:
                                    # No files left, delete entire org graph
                                    await conn.execute(
                                        "DELETE FROM noderag_graphs WHERE org_id = $1",
                                        org_id
                                    )
                                    logger.info(f"🗑️ Deleted entire org graph for {org_id} (no files remaining)")
                                    graph_updated = True
                                else:
                                    # Re-serialize and update org graph
                                    updated_graph_data = pickle.dumps({
                                        'graph': graph_manager.graph,
                                        'entity_index': dict(graph_manager.entity_index),
                                        'community_assignments': graph_manager.community_assignments
                                    })

                                    # Get updated stats
                                    updated_stats = graph_manager.get_org_stats(processed_files)
                                    new_version = org_graph_row['version'] + 1

                                    # Update org graph
                                    await conn.execute("""
                                        UPDATE noderag_graphs
                                        SET graph_data = $1,
                                            processed_files = $2,
                                            version = $3,
                                            stats = $4,
                                            last_incremental_update = CURRENT_TIMESTAMP,
                                            updated_at = CURRENT_TIMESTAMP
                                        WHERE org_id = $5
                                    """, updated_graph_data, json.dumps(processed_files),
                                         new_version, json.dumps(updated_stats), org_id)

                                    logger.info(f"✅ Updated org graph: removed {nodes_removed} nodes from file {file_id}")
                                    graph_updated = True
                            else:
                                logger.warning(f"File {file_id} not found in processed_files for org {org_id}")

                        except Exception as graph_error:
                            logger.error(f"Error updating org graph: {graph_error}")
                            # Continue anyway - embeddings were deleted

                    delete_time = time.time() - delete_start
                    logger.info(f"🗑️ Deleted {embeddings_deleted} embeddings and {nodes_removed} graph nodes for file_id={file_id} in {delete_time:.2f}s")

                    return {
                        "success": True,
                        "deleted_count": embeddings_deleted,
                        "nodes_removed": nodes_removed,
                        "graph_updated": graph_updated,
                        "delete_time_seconds": delete_time
                    }

            finally:
                await self._release_connection(conn)

        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            import traceback
            traceback.print_exc()
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
    
    # ==================== ORG-LEVEL GRAPH METHODS ====================
    
    async def _load_org_graph_async(self, org_id: str) -> Optional[Dict]:
        """Load existing org graph with metadata"""
        try:
            conn = await self._get_connection()
            try:
                query = """
                    SELECT org_id, user_id, graph_data, stats, processed_files, 
                           version, last_file_added, last_incremental_update, created_at
                    FROM noderag_graphs 
                    WHERE org_id = $1
                """
                row = await conn.fetchrow(query, org_id)
                
                if row:
                    return {
                        'org_id': row['org_id'],
                        'user_id': row['user_id'],
                        'graph_data': row['graph_data'],
                        'stats': json.loads(row['stats']) if row['stats'] else {},
                        'processed_files': row['processed_files'] if row['processed_files'] else [],
                        'version': row['version'],
                        'last_file_added': row['last_file_added'],
                        'last_incremental_update': row['last_incremental_update'],
                        'created_at': row['created_at']
                    }
                return None
                
            finally:
                await self._release_connection(conn)
                
        except Exception as e:
            logger.error(f"Error loading org graph for {org_id}: {e}")
            return None
    
    def load_org_graph(self, org_id: str) -> Optional[Dict]:
        """Sync wrapper for loading org graph"""
        return run_async_safe(self._load_org_graph_async(org_id))
    
    async def _store_org_graph_async(self, org_id: str, graph_data: bytes, 
                                   processed_files: List[str], version: int, 
                                   last_file_added: str, stats: Dict, user_id: str) -> Dict:
        """Store updated org graph with file tracking"""
        start_time = time.time()
        
        try:
            await self._ensure_tables_exist()
            conn = await self._get_connection()
            
            try:
                async with conn.transaction():
                    # Store/update org graph
                    await conn.execute("""
                        INSERT INTO noderag_graphs 
                        (org_id, user_id, graph_data, stats, processed_files, version, 
                         last_file_added, last_incremental_update)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
                        ON CONFLICT (org_id) 
                        DO UPDATE SET 
                            graph_data = EXCLUDED.graph_data,
                            stats = EXCLUDED.stats,
                            processed_files = EXCLUDED.processed_files,
                            version = EXCLUDED.version,
                            last_file_added = EXCLUDED.last_file_added,
                            last_incremental_update = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                    """, org_id, user_id, graph_data, json.dumps(stats), 
                         json.dumps(processed_files), version, last_file_added)
                    
                    # Update file processing status
                    await conn.execute("""
                        INSERT INTO noderag_file_processing 
                        (org_id, file_id, processing_status, processing_time, completed_at)
                        VALUES ($1, $2, 'completed', $3, CURRENT_TIMESTAMP)
                        ON CONFLICT (org_id, file_id)
                        DO UPDATE SET 
                            processing_status = 'completed',
                            processing_time = EXCLUDED.processing_time,
                            completed_at = CURRENT_TIMESTAMP
                    """, org_id, last_file_added, time.time() - start_time)
                    
                    logger.info(f"✅ Stored org graph for {org_id}: version {version}, files: {len(processed_files)}")
                    
                    return {
                        "success": True,
                        "org_id": org_id,
                        "version": version,
                        "processed_files": processed_files,
                        "storage_time_seconds": time.time() - start_time
                    }
            
            finally:
                await self._release_connection(conn)
                
        except Exception as e:
            logger.error(f"Error storing org graph for {org_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "storage_time_seconds": time.time() - start_time
            }
    
    def store_org_graph(self, org_id: str, graph_data: bytes, processed_files: List[str], 
                       version: int, last_file_added: str, stats: Dict, user_id: str) -> Dict:
        """Sync wrapper for storing org graph"""
        return run_async_safe(self._store_org_graph_async(
            org_id, graph_data, processed_files, version, last_file_added, stats, user_id
        ))
    
    async def _check_file_already_processed_async(self, org_id: str, file_id: str) -> bool:
        """Check if file already exists in org graph"""
        try:
            existing_graph = await self._load_org_graph_async(org_id)
            if existing_graph:
                return file_id in existing_graph.get('processed_files', [])
            return False
        except Exception as e:
            logger.error(f"Error checking file status for {org_id}/{file_id}: {e}")
            return False
    
    def check_file_already_processed(self, org_id: str, file_id: str) -> bool:
        """Sync wrapper for checking file processing status"""
        return run_async_safe(self._check_file_already_processed_async(org_id, file_id))
    
    async def _get_org_processed_files_async(self, org_id: str) -> List[str]:
        """Get list of files already processed for org"""
        try:
            existing_graph = await self._load_org_graph_async(org_id)
            if existing_graph:
                return existing_graph.get('processed_files', [])
            return []
        except Exception as e:
            logger.error(f"Error getting processed files for {org_id}: {e}")
            return []
    
    def get_org_processed_files(self, org_id: str) -> List[str]:
        """Sync wrapper for getting processed files"""
        return run_async_safe(self._get_org_processed_files_async(org_id))
    
    # ==================== SYNCHRONOUS ORG OPERATIONS ====================
    
    def _get_sync_connection(self):
        """Get synchronous database connection using psycopg2"""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not available for sync operations")
        
        # Convert asyncpg URL to psycopg2 format
        db_url = self.db_url
        if db_url.startswith('postgresql://'):
            db_url = db_url.replace('postgresql://', 'postgres://')
        
        return psycopg2.connect(db_url)
    
    def load_org_graph_sync(self, org_id: str) -> Optional[Dict]:
        """Load org graph using synchronous connection"""
        try:
            with self._get_sync_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT org_id, user_id, graph_data, stats, processed_files, 
                               version, last_file_added, last_incremental_update, created_at
                        FROM noderag_graphs 
                        WHERE org_id = %s
                    """, (org_id,))
                    
                    row = cur.fetchone()
                    if row:
                        return {
                            'org_id': row['org_id'],
                            'user_id': row['user_id'],
                            'graph_data': bytes(row['graph_data']) if row['graph_data'] else None,
                            'stats': row['stats'] if row['stats'] else {},
                            'processed_files': row['processed_files'] if row['processed_files'] else [],
                            'version': row['version'],
                            'last_file_added': row['last_file_added'],
                            'last_incremental_update': row['last_incremental_update'],
                            'created_at': row['created_at']
                        }
                    return None
                    
        except Exception as e:
            logger.error(f"Error loading org graph for {org_id} (sync): {e}")
            return None
    
    def store_org_graph_sync(self, org_id: str, graph_data: bytes, processed_files: List[str], 
                           version: int, last_file_added: str, stats: Dict, user_id: str) -> Dict:
        """Store org graph using synchronous connection"""
        start_time = time.time()
        
        try:
            with self._get_sync_connection() as conn:
                with conn.cursor() as cur:
                    # Store/update org graph (include file_id as NULL for org-level storage)
                    cur.execute("""
                        INSERT INTO noderag_graphs 
                        (org_id, user_id, file_id, graph_data, stats, processed_files, version, 
                         last_file_added, last_incremental_update)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (org_id) 
                        DO UPDATE SET 
                            graph_data = EXCLUDED.graph_data,
                            stats = EXCLUDED.stats,
                            processed_files = EXCLUDED.processed_files,
                            version = EXCLUDED.version,
                            last_file_added = EXCLUDED.last_file_added,
                            last_incremental_update = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                    """, (org_id, user_id, None, psycopg2.Binary(graph_data), 
                          json.dumps(stats), json.dumps(processed_files), version, last_file_added))
                    
                    # Update file processing status
                    cur.execute("""
                        INSERT INTO noderag_file_processing 
                        (org_id, file_id, processing_status, processing_time, completed_at)
                        VALUES (%s, %s, 'completed', %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (org_id, file_id)
                        DO UPDATE SET 
                            processing_status = 'completed',
                            processing_time = EXCLUDED.processing_time,
                            completed_at = CURRENT_TIMESTAMP
                    """, (org_id, last_file_added, time.time() - start_time))
                    
                    conn.commit()
                    
                    logger.info(f"✅ Stored org graph for {org_id} (sync): version {version}, files: {len(processed_files)}")
                    
                    return {
                        "success": True,
                        "org_id": org_id,
                        "version": version,
                        "processed_files": processed_files,
                        "storage_time_seconds": time.time() - start_time
                    }
                    
        except Exception as e:
            logger.error(f"Error storing org graph for {org_id} (sync): {e}")
            return {
                "success": False,
                "error": str(e),
                "storage_time_seconds": time.time() - start_time
            }
    # ==================== VECTOR SIMILARITY SEARCH METHODS ====================
    
    def vector_similarity_search_sync(self, 
                                      query_embedding: list, 
                                      org_id: str, 
                                      k: int = 10,
                                      similarity_threshold: float = 0.7) -> list:
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
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
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
                            "node_id": row["node_id"],
                            "node_type": row["node_type"],
                            "content": row["content"],
                            "similarity_score": float(row["similarity_score"]),
                            "metadata": row["graph_metadata"] or {}
                        })
                    
                    logger.debug(f"Vector search found {len(results)} results for org {org_id}")
                    return results
                    
        except Exception as e:
            logger.error(f"Error in vector similarity search for org {org_id}: {e}")
            return []
