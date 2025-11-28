#!/usr/bin/env python3
"""
Neo4j Storage for NodeRAG - handles graph and embedding storage in Neo4j
High-performance implementation with token tracking and fast processing
"""

import os
import json
import pickle
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, TransientError
import time

from ..config.settings import Config
from ..graph.node_types import NodeType
from ..llm.llm_service import LLMService

logger = logging.getLogger(__name__)

class Neo4jStorage:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "neo4j+s://4b2d4f30.databases.neo4j.io")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "1DyQZbgG4Cr1mQMdBZtZMEC0HpehqmwBEj6puoSsNGU")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        if not all([self.uri, self.username, self.password]):
            raise ValueError("Neo4j connection parameters (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) not found")
        
        self.driver: Optional[Driver] = None
        self.llm_service = LLMService()
        self._initialize_driver()
        
    def _initialize_driver(self):
        """Initialize Neo4j driver with optimized settings for performance"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password),
                max_connection_lifetime=3600,  # 1 hour
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("✅ Neo4j connection established successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise
    
    def _get_session(self):
        """Get Neo4j session with optimized settings"""
        if not self.driver:
            self._initialize_driver()
        return self.driver.session(database=self.database, default_access_mode="WRITE")
    
    def _execute_query(self, query: str, **params):
        """Execute query using modern execute_query method"""
        if not self.driver:
            self._initialize_driver()
        return self.driver.execute_query(query, database_=self.database, **params)
    
    def _ensure_constraints_and_indexes(self):
        """Create constraints and indexes for optimal performance"""
        with self._get_session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT node_embedding_id IF NOT EXISTS FOR (n:NodeEmbedding) REQUIRE n.node_id IS UNIQUE",
                "CREATE CONSTRAINT graph_data_composite IF NOT EXISTS FOR (g:GraphData) REQUIRE (g.org_id, g.file_id) IS UNIQUE"
            ]
            
            # Create indexes for fast queries
            indexes = [
                "CREATE INDEX node_embedding_org_file IF NOT EXISTS FOR (n:NodeEmbedding) ON (n.org_id, n.file_id)",
                "CREATE INDEX node_embedding_type IF NOT EXISTS FOR (n:NodeEmbedding) ON n.node_type",
                "CREATE INDEX node_embedding_created IF NOT EXISTS FOR (n:NodeEmbedding) ON n.created_at",
                "CREATE INDEX graph_data_org IF NOT EXISTS FOR (g:GraphData) ON g.org_id",
                "CREATE INDEX graph_data_created IF NOT EXISTS FOR (g:GraphData) ON g.created_at"
            ]
            
            all_commands = constraints + indexes
            
            for command in all_commands:
                try:
                    session.run(command)
                except Exception as e:
                    # Constraints/indexes might already exist, which is fine
                    logger.debug(f"Command might already exist: {command} - {e}")
            
            logger.info("✅ Neo4j constraints and indexes ensured")
    
    def store_noderag_data(self, org_id: str, file_id: str, user_id: str, pipeline, 
                          input_tokens: int = 0, output_tokens: int = 0, api_calls: int = 0) -> Dict[str, Any]:
        """Store NodeRAG graph and embeddings data with token tracking"""
        start_time = time.time()
        total_tokens = input_tokens + output_tokens
        
        try:
            self._ensure_constraints_and_indexes()
            
            with self._get_session() as session:
                stored_embeddings = 0
                stored_graphs = 0
                
                # Store graph metadata first (separate transaction)
                with session.begin_transaction() as tx:
                    
                    # 1. Store graph data with token tracking
                    print("🔄 Serializing graph data...")
                    graph_data_binary = pickle.dumps(pipeline.graph_manager.graph)
                    graph_stats = pipeline.graph_manager.get_stats()
                    print(f"✅ Graph serialized ({len(graph_data_binary)} bytes)")
                    
                    print("🔄 Storing graph metadata...")
                    tx.run("""
                        MERGE (g:GraphData {org_id: $org_id, file_id: $file_id})
                        SET g.user_id = $user_id,
                            g.stats = $stats,
                            g.input_tokens = $input_tokens,
                            g.output_tokens = $output_tokens,
                            g.total_tokens = $total_tokens,
                            g.api_calls = $api_calls,
                            g.updated_at = datetime(),
                            g.created_at = coalesce(g.created_at, datetime())
                    """, 
                        org_id=org_id, file_id=file_id, user_id=user_id,
                        stats=json.dumps(graph_stats),
                        input_tokens=input_tokens, output_tokens=output_tokens,
                        total_tokens=total_tokens, api_calls=api_calls
                    )
                    print("✅ Graph metadata stored")
                    
                    stored_graphs = 1
                    logger.info(f"✅ Stored graph data for file_id={file_id}")
                
                # Delete existing embeddings for this file (separate transaction)
                with session.begin_transaction() as tx:
                    print("🧹 Cleaning existing embeddings...")
                    result = tx.run("""
                        MATCH (n:NodeEmbedding {org_id: $org_id, file_id: $file_id})
                        DELETE n
                        RETURN count(n) as deleted
                    """, org_id=org_id, file_id=file_id)
                    
                    deleted_count = result.single()["deleted"]
                    print(f"🗑️ Deleted {deleted_count} existing embeddings")
                
                # Collect nodes for storage (outside transaction)
                print("🔄 Collecting nodes for storage...")
                all_nodes = []
                for node_type in [NodeType.SEMANTIC, NodeType.ENTITY, NodeType.RELATIONSHIP, 
                                 NodeType.ATTRIBUTE, NodeType.HIGH_LEVEL, NodeType.OVERVIEW]:
                    try:
                        nodes = pipeline.graph_manager.get_nodes_by_type(node_type)
                        all_nodes.extend(nodes)
                        print(f"  ✅ Collected {len(nodes)} {node_type.value} nodes")
                    except Exception as e:
                        print(f"  ⚠️  Could not get {node_type.value} nodes: {e}")
                        logger.warning(f"Could not get nodes of type {node_type}: {e}")
                    
                print(f"📊 Found {len(all_nodes)} total nodes to store")
                logger.info(f"📊 Found {len(all_nodes)} nodes to store")
                
                # Batch insert embeddings for performance (separate transactions)
                batch_size = 25  # Small batches to avoid timeouts
                print(f"🔄 Processing {len(all_nodes)} nodes in batches of {batch_size}...")
                
                for i in range(0, len(all_nodes), batch_size):
                    batch_nodes = all_nodes[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(all_nodes) + batch_size - 1) // batch_size
                    print(f"  🔄 Processing batch {batch_num}/{total_batches} ({len(batch_nodes)} nodes)...")
                    
                    # Process each batch in separate transaction
                    with session.begin_transaction() as tx:
                        embedding_params = []
                        
                        for idx, node in enumerate(batch_nodes):
                            if hasattr(node, 'embeddings') and node.embeddings is not None:
                                graph_metadata = {
                                    'node_metadata': node.metadata if hasattr(node, 'metadata') else {},
                                    'node_type_value': node.type.value if hasattr(node.type, 'value') else str(node.type)
                                }
                                
                                embedding_params.append({
                                    'node_id': node.id,
                                    'node_type': node.type.value if hasattr(node.type, 'value') else str(node.type),
                                    'content': node.content[:2000],  # Limit content length
                                    'embedding': node.embeddings,
                                    'org_id': org_id,
                                    'file_id': file_id,
                                    'user_id': user_id,
                                    'chunk_index': stored_embeddings + idx,
                                    'graph_metadata': json.dumps(graph_metadata),
                                    'input_tokens': input_tokens // len(all_nodes) if all_nodes else 0,
                                    'output_tokens': output_tokens // len(all_nodes) if all_nodes else 0,
                                    'total_tokens': total_tokens // len(all_nodes) if all_nodes else 0,
                                    'api_calls': api_calls // len(all_nodes) if all_nodes else 0
                                })
                        
                        if embedding_params:
                            print(f"    💾 Storing {len(embedding_params)} embeddings...")
                            tx.run("""
                                UNWIND $batch_params AS params
                                MERGE (n:NodeEmbedding {node_id: params.node_id})
                                SET n += params,
                                    n.created_at = coalesce(n.created_at, datetime()),
                                    n.updated_at = datetime()
                            """, batch_params=embedding_params)
                            
                            stored_embeddings += len(embedding_params)
                            print(f"    ✅ Batch {batch_num} stored ({len(embedding_params)} embeddings)")
                        else:
                            print(f"    ⚠️  Batch {batch_num} had no valid embeddings")
                    
                    logger.info(f"✅ Stored {stored_embeddings} embeddings for file_id={file_id}")
                
                processing_time = time.time() - start_time
                logger.info(f"⚡ Storage completed in {processing_time:.2f}s")
                
                return {
                    "success": True,
                    "embeddings_stored": stored_embeddings,
                    "graphs_stored": stored_graphs,
                    "graph_nodes": len(all_nodes),
                    "processing_time": processing_time,
                    "tokens_tracked": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "api_calls": api_calls
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Error storing NodeRAG data: {e}")
            return {
                "success": False,
                "error": str(e),
                "embeddings_stored": 0,
                "graphs_stored": 0
            }
    
    def search_noderag_data(self, org_id: str, query: str, top_k: int = 10, filters: Dict = None) -> List[Dict]:
        """Search NodeRAG embeddings using cosine similarity"""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embeddings = self.llm_service.get_embeddings([query])
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            
            with self._get_session() as session:
                # Build dynamic query with filters
                where_conditions = ["n.org_id = $org_id"]
                params = {"org_id": org_id, "query_embedding": query_embedding, "top_k": top_k}
                
                if filters:
                    if 'file_id' in filters:
                        where_conditions.append("n.file_id = $file_id")
                        params["file_id"] = filters['file_id']
                    
                    if 'node_type' in filters:
                        where_conditions.append("n.node_type = $node_type")
                        params["node_type"] = filters['node_type']
                
                where_clause = " AND ".join(where_conditions)
                
                # Use vector similarity search with cosine similarity
                cypher_query = f"""
                    MATCH (n:NodeEmbedding)
                    WHERE {where_clause}
                    WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS similarity_score
                    ORDER BY similarity_score DESC
                    LIMIT $top_k
                    RETURN n.node_id, n.node_type, n.content, n.file_id, n.user_id,
                           n.chunk_index, n.graph_metadata, n.input_tokens, n.output_tokens,
                           n.total_tokens, n.api_calls, similarity_score
                """
                
                result = session.run(cypher_query, **params)
                
                results = []
                for record in result:
                    results.append({
                        "node_id": record["n.node_id"],
                        "node_type": record["n.node_type"],
                        "content": record["n.content"],
                        "file_id": record["n.file_id"],
                        "user_id": record["n.user_id"],
                        "chunk_index": record["n.chunk_index"],
                        "metadata": json.loads(record["n.graph_metadata"]) if record["n.graph_metadata"] else {},
                        "similarity_score": float(record["similarity_score"]),
                        "token_info": {
                            "input_tokens": record["n.input_tokens"] or 0,
                            "output_tokens": record["n.output_tokens"] or 0,
                            "total_tokens": record["n.total_tokens"] or 0,
                            "api_calls": record["n.api_calls"] or 0
                        }
                    })
                
                search_time = time.time() - start_time
                logger.info(f"🔍 Found {len(results)} search results in {search_time:.2f}s for query: '{query[:50]}...'")
                return results
                
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            # Fallback to basic similarity without GDS functions
            return self._fallback_search(org_id, query, query_embedding, top_k, filters)
    
    def _fallback_search(self, org_id: str, query: str, query_embedding: List[float], 
                        top_k: int = 10, filters: Dict = None) -> List[Dict]:
        """Fallback search without GDS functions"""
        try:
            with self._get_session() as session:
                where_conditions = ["n.org_id = $org_id"]
                params = {"org_id": org_id, "top_k": top_k}
                
                if filters:
                    if 'file_id' in filters:
                        where_conditions.append("n.file_id = $file_id")
                        params["file_id"] = filters['file_id']
                    
                    if 'node_type' in filters:
                        where_conditions.append("n.node_type = $node_type")
                        params["node_type"] = filters['node_type']
                
                where_clause = " AND ".join(where_conditions)
                
                # Basic text similarity using CONTAINS for now
                cypher_query = f"""
                    MATCH (n:NodeEmbedding)
                    WHERE {where_clause} AND toLower(n.content) CONTAINS toLower($query_text)
                    RETURN n.node_id, n.node_type, n.content, n.file_id, n.user_id,
                           n.chunk_index, n.graph_metadata, n.input_tokens, n.output_tokens,
                           n.total_tokens, n.api_calls
                    LIMIT $top_k
                """
                
                params["query_text"] = query
                result = session.run(cypher_query, **params)
                
                results = []
                for record in result:
                    results.append({
                        "node_id": record["n.node_id"],
                        "node_type": record["n.node_type"],
                        "content": record["n.content"],
                        "file_id": record["n.file_id"],
                        "user_id": record["n.user_id"],
                        "chunk_index": record["n.chunk_index"],
                        "metadata": json.loads(record["n.graph_metadata"]) if record["n.graph_metadata"] else {},
                        "similarity_score": 0.8,  # Default score for text match
                        "token_info": {
                            "input_tokens": record["n.input_tokens"] or 0,
                            "output_tokens": record["n.output_tokens"] or 0,
                            "total_tokens": record["n.total_tokens"] or 0,
                            "api_calls": record["n.api_calls"] or 0
                        }
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"❌ Fallback search error: {e}")
            return []
    
    def delete_file_data(self, org_id: str, file_id: str) -> Dict[str, Any]:
        """Delete all data for a specific file"""
        try:
            with self._get_session() as session:
                with session.begin_transaction() as tx:
                    # Delete embeddings
                    embedding_result = tx.run("""
                        MATCH (n:NodeEmbedding {org_id: $org_id, file_id: $file_id})
                        DELETE n
                        RETURN count(n) as deleted_embeddings
                    """, org_id=org_id, file_id=file_id)
                    
                    # Delete graph data
                    graph_result = tx.run("""
                        MATCH (g:GraphData {org_id: $org_id, file_id: $file_id})
                        DELETE g
                        RETURN count(g) as deleted_graphs
                    """, org_id=org_id, file_id=file_id)
                    
                    embedding_record = embedding_result.single()
                    graph_record = graph_result.single()
                    
                    embeddings_deleted = embedding_record["deleted_embeddings"] if embedding_record else 0
                    graphs_deleted = graph_record["deleted_graphs"] if graph_record else 0
                    
                    logger.info(f"🗑️ Deleted {embeddings_deleted} embeddings and {graphs_deleted} graphs for file_id={file_id}")
                    
                    return {
                        "success": True,
                        "deleted_count": embeddings_deleted,
                        "graphs_deleted": graphs_deleted
                    }
                    
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_count": 0
            }
    
    def get_file_stats(self, org_id: str, file_id: str = None) -> Dict[str, Any]:
        """Get statistics for files including token usage"""
        try:
            with self._get_session() as session:
                if file_id:
                    # Get stats for specific file
                    result = session.run("""
                        MATCH (n:NodeEmbedding {org_id: $org_id, file_id: $file_id})
                        RETURN count(n) as embedding_count,
                               sum(n.input_tokens) as total_input_tokens,
                               sum(n.output_tokens) as total_output_tokens,
                               sum(n.total_tokens) as total_tokens,
                               sum(n.api_calls) as total_api_calls
                    """, org_id=org_id, file_id=file_id)
                    
                    graph_result = session.run("""
                        MATCH (g:GraphData {org_id: $org_id, file_id: $file_id})
                        RETURN g.stats as graph_stats,
                               g.input_tokens as graph_input_tokens,
                               g.output_tokens as graph_output_tokens,
                               g.total_tokens as graph_total_tokens,
                               g.api_calls as graph_api_calls
                    """, org_id=org_id, file_id=file_id)
                    
                    record = result.single()
                    graph_record = graph_result.single()
                    
                    return {
                        "org_id": org_id,
                        "file_id": file_id,
                        "embedding_count": record["embedding_count"] if record else 0,
                        "token_usage": {
                            "embedding_input_tokens": record["total_input_tokens"] if record else 0,
                            "embedding_output_tokens": record["total_output_tokens"] if record else 0,
                            "embedding_total_tokens": record["total_tokens"] if record else 0,
                            "embedding_api_calls": record["total_api_calls"] if record else 0,
                            "graph_input_tokens": graph_record["graph_input_tokens"] if graph_record else 0,
                            "graph_output_tokens": graph_record["graph_output_tokens"] if graph_record else 0,
                            "graph_total_tokens": graph_record["graph_total_tokens"] if graph_record else 0,
                            "graph_api_calls": graph_record["graph_api_calls"] if graph_record else 0
                        },
                        "graph_stats": json.loads(graph_record["graph_stats"]) if graph_record and graph_record["graph_stats"] else {}
                    }
                else:
                    # Get stats for all files in org
                    result = session.run("""
                        MATCH (n:NodeEmbedding {org_id: $org_id})
                        RETURN count(n) as total_embeddings,
                               sum(n.input_tokens) as total_input_tokens,
                               sum(n.output_tokens) as total_output_tokens,
                               sum(n.total_tokens) as total_tokens,
                               sum(n.api_calls) as total_api_calls
                    """, org_id=org_id)
                    
                    graph_count_result = session.run("""
                        MATCH (g:GraphData {org_id: $org_id})
                        RETURN count(g) as total_graphs,
                               sum(g.input_tokens) as graph_input_tokens,
                               sum(g.output_tokens) as graph_output_tokens,
                               sum(g.total_tokens) as graph_total_tokens,
                               sum(g.api_calls) as graph_api_calls
                    """, org_id=org_id)
                    
                    files_result = session.run("""
                        MATCH (n:NodeEmbedding {org_id: $org_id})
                        RETURN n.file_id as file_id, count(n) as embedding_count,
                               sum(n.input_tokens) as input_tokens,
                               sum(n.output_tokens) as output_tokens,
                               sum(n.total_tokens) as total_tokens,
                               sum(n.api_calls) as api_calls
                        ORDER BY embedding_count DESC
                    """, org_id=org_id)
                    
                    record = result.single()
                    graph_record = graph_count_result.single()
                    
                    files = []
                    for file_record in files_result:
                        files.append({
                            "file_id": file_record["file_id"],
                            "embedding_count": file_record["embedding_count"],
                            "token_usage": {
                                "input_tokens": file_record["input_tokens"],
                                "output_tokens": file_record["output_tokens"], 
                                "total_tokens": file_record["total_tokens"],
                                "api_calls": file_record["api_calls"]
                            }
                        })
                    
                    return {
                        "org_id": org_id,
                        "total_embeddings": record["total_embeddings"] if record else 0,
                        "total_graphs": graph_record["total_graphs"] if graph_record else 0,
                        "total_token_usage": {
                            "embedding_input_tokens": record["total_input_tokens"] if record else 0,
                            "embedding_output_tokens": record["total_output_tokens"] if record else 0,
                            "embedding_total_tokens": record["total_tokens"] if record else 0,
                            "embedding_api_calls": record["total_api_calls"] if record else 0,
                            "graph_input_tokens": graph_record["graph_input_tokens"] if graph_record else 0,
                            "graph_output_tokens": graph_record["graph_output_tokens"] if graph_record else 0,
                            "graph_total_tokens": graph_record["graph_total_tokens"] if graph_record else 0,
                            "graph_api_calls": graph_record["graph_api_calls"] if graph_record else 0
                        },
                        "files": files
                    }
                    
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {"error": str(e)}
    
    def inspect_all_data(self) -> Dict[str, Any]:
        """Inspect all data in the database for debugging"""
        try:
            with self._get_session() as session:
                # Get summary of all data with token information
                result = session.run("""
                    MATCH (n:NodeEmbedding)
                    RETURN n.file_id as file_id, n.org_id as org_id, n.node_type as node_type,
                           count(n) as count, sum(n.total_tokens) as total_tokens,
                           sum(n.api_calls) as api_calls,
                           min(n.created_at) as first_created, max(n.created_at) as last_created
                    ORDER BY last_created DESC
                    LIMIT 50
                """)
                
                results = []
                for record in result:
                    results.append({
                        "file_id": record["file_id"],
                        "org_id": record["org_id"],
                        "node_type": record["node_type"],
                        "count": record["count"],
                        "total_tokens": record["total_tokens"] or 0,
                        "api_calls": record["api_calls"] or 0,
                        "first_created": str(record["first_created"]) if record["first_created"] else None,
                        "last_created": str(record["last_created"]) if record["last_created"] else None
                    })
                
                return {
                    "success": True,
                    "results": results,
                    "total_groups": len(results)
                }
                
        except Exception as e:
            logger.error(f"❌ Inspect error: {e}")
            return {"success": False, "error": str(e)}
    
    def store_api_usage(self, org_id: str, user_id: str, endpoint: str, 
                       input_tokens: int = 0, output_tokens: int = 0, 
                       total_tokens: int = 0, api_calls: int = 0, 
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store API usage metrics for analytics and billing"""
        try:
            with self._get_session() as session:
                # Create API usage node with comprehensive tracking
                session.run("""
                    CREATE (u:ApiUsage {
                        org_id: $org_id,
                        user_id: $user_id,
                        endpoint: $endpoint,
                        input_tokens: $input_tokens,
                        output_tokens: $output_tokens,
                        total_tokens: $total_tokens,
                        api_calls: $api_calls,
                        metadata: $metadata,
                        timestamp: datetime(),
                        created_at: datetime()
                    })
                """, 
                    org_id=org_id,
                    user_id=user_id,
                    endpoint=endpoint,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    api_calls=api_calls,
                    metadata=json.dumps(metadata) if metadata else "{}"
                )
                
                logger.debug(f"📊 Stored API usage: {endpoint} - tokens: {total_tokens}, calls: {api_calls}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to store API usage: {e}")
            return False
    
    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("✅ Neo4j connection closed")