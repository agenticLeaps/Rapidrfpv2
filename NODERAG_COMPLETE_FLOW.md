# NodeRAG Complete Flow Documentation

## Overview
This document provides a comprehensive guide to the NodeRAG API flow, covering document processing and response generation with detailed code examples and architectural explanations.

## Architecture Summary
- **Org-Level Graph Management**: Single unified graph per organization
- **Incremental Processing**: New files enhance existing graphs
- **Database-First Storage**: PostgreSQL with pgvector extension
- **Multi-Signal Retrieval**: Vector search + Exact matching + PageRank
- **Production Optimized**: No disk storage, concurrent protection

---

## 🚀 1. PROCESS-DOCUMENT FLOW

### API Endpoint
```
POST /api/v1/process-document
```

### Request Format
```json
{
    "org_id": "organization-uuid",
    "file_id": "file-uuid", 
    "user_id": "user-uuid",
    "chunks": [
        {
            "content": "Document text content...",
            "metadata": {"page": 1, "section": "intro"}
        }
    ],
    "callback_url": "https://your-app.com/webhook" // Optional
}
```

### Step 1: Request Validation & Setup
```python
@app.route("/api/v1/process-document", methods=["POST"])
def process_document():
    data = request.get_json()
    
    # Required fields validation
    required_fields = ["org_id", "file_id", "user_id", "chunks"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    org_id = data["org_id"]
    file_id = data["file_id"] 
    user_id = data["user_id"]
    chunks = data["chunks"]  # List of document chunks
    callback_url = data.get("callback_url")
    
    # Start background processing
    threading.Thread(target=process_document_pipeline, 
                    args=(org_id, file_id, user_id, chunks, callback_url), 
                    daemon=True).start()
    
    return jsonify({
        "message": "Document processing started",
        "file_id": file_id,
        "status": "processing"
    }), 202
```

### Step 2: Concurrent Protection with Org-Level Locking
```python
def process_document_pipeline(org_id: str, file_id: str, user_id: str, 
                             chunks: List[Dict], callback_url: str = None):
    """Org-level incremental graph processing with concurrent protection"""
    
    # 🔒 CONCURRENT PROTECTION: Use org-level lock
    org_lock = get_or_create_org_lock(org_id)
    
    with org_lock:  # Ensures only one file processes per org at a time
        logger.info(f"🔒 Acquired org lock for {org_id}, processing file {file_id}")
        # Processing happens here...
```

### Step 3: Graph Discovery - Check Existing Org Graph
```python
# 🔍 STEP 1: Check if org already has a graph
storage = noderag_service.get_neon_storage()
existing_graph = storage.load_org_graph_sync(org_id)

if existing_graph:
    logger.info(f"📊 Found existing org graph: {len(existing_graph['processed_files'])} files processed")
    
    # Check if this file was already processed
    if file_id in existing_graph['processed_files']:
        logger.info(f"⚠️ File {file_id} already processed for org {org_id}")
        # Skip processing - return early
        return
    
    # Load existing graph into pipeline for incremental processing
    pipeline = noderag_service.get_pipeline()
    pipeline.set_incremental_mode(org_id, file_id, existing_graph['graph_data'])
    is_incremental = True
    
else:
    logger.info(f"🆕 Creating first graph for org {org_id}")
    # Creating first graph for this organization
    pipeline = noderag_service.get_pipeline()
    pipeline.set_incremental_mode(org_id, file_id)  # Fresh pipeline
    is_incremental = False
```

### Step 4: Document Chunk Processing
```python
# Convert chunks to NodeRAG format
noderag_chunks = []
for i, chunk_data in enumerate(chunks):
    chunk_text = chunk_data.get('content', chunk_data.get('text', ''))
    if chunk_text.strip():
        noderag_chunk = DocumentChunk(
            content=chunk_text,
            chunk_index=i,
            start_char=0,
            end_char=len(chunk_text),
            metadata={
                'file_id': file_id,
                'org_id': org_id,
                'user_id': user_id,
                'chunk_index': i,
                'original_metadata': chunk_data.get('metadata', {})
            },
            token_count=len(chunk_text.split())
        )
        noderag_chunks.append(noderag_chunk)

processed_doc = ProcessedDocument(
    chunks=noderag_chunks,
    metadata={
        'file_id': file_id,
        'org_id': org_id, 
        'user_id': user_id,
        'total_chunks': len(noderag_chunks),
        'incremental_mode': is_incremental
    },
    total_tokens=sum(chunk.token_count for chunk in noderag_chunks)
)
```

---

## 🧠 Phase 1: Graph Decomposition

### Step 5: Extract Base Nodes (T, S, N, R)
```python
# Phase 1: Graph Decomposition
logger.info(f"🔄 Phase 1: Graph Decomposition ({'Incremental' if is_incremental else 'Initial'})")
result = pipeline._phase_1_decomposition(processed_doc)
if not result['success']:
    raise Exception(f"Phase 1 failed: {result.get('error')}")
```

#### Node Creation Process:
```python
def _process_chunk(self, chunk: DocumentChunk) -> bool:
    try:
        chunk_id = str(uuid.uuid4())
        
        # 📝 Create Text Node (T)
        text_node = Node(
            id=f"T_{chunk_id}",
            type=NodeType.TEXT,
            content=chunk.content,
            metadata={
                **chunk.metadata,
                'node_type': 'text',
                'chunk_id': chunk_id
            }
        )
        self.graph_manager.add_node(text_node)
        
        # 🤖 Extract content using LLM
        extraction_result = self.llm_service.extract_all_from_chunk(chunk.content)
        
        if not extraction_result.success:
            logger.warning(f"LLM extraction failed for chunk {chunk.chunk_index}")
            return False
            
        # 🔗 Create Semantic Unit nodes (S) with dynamic numbering
        existing_semantic_count = len([node_id for node_id in self.graph_manager.graph.nodes() 
                                     if node_id.startswith(f"S_{chunk_id}_")])
        
        for i, semantic_unit in enumerate(extraction_result.semantic_units):
            semantic_index = existing_semantic_count + i  # Dynamic increment
            semantic_id = f"S_{chunk_id}_{semantic_index}"
            semantic_node = Node(
                id=semantic_id,
                type=NodeType.SEMANTIC,
                content=semantic_unit,
                metadata={
                    **chunk.metadata,
                    'chunk_id': chunk_id,
                    'semantic_index': semantic_index,
                    'source_file': chunk.metadata.get('file_path'),
                    'node_type': 'semantic'
                }
            )
            self.graph_manager.add_node(semantic_node)
            
            # Link to text node
            edge = Edge(
                source=text_node.id,
                target=semantic_id,
                relationship_type="contains_semantic_unit"
            )
            self.graph_manager.add_edge(edge)
            
        # 👤 Create Entity nodes (N) with dynamic numbering  
        existing_entity_count = len([node_id for node_id in self.graph_manager.graph.nodes() 
                                   if node_id.startswith(f"N_{chunk_id}_")])
        
        for i, entity in enumerate(extraction_result.entities):
            entity_index = existing_entity_count + i  # Dynamic increment
            entity_id = f"N_{chunk_id}_{entity_index}"
            entity_node = Node(
                id=entity_id,
                type=NodeType.ENTITY,
                content=entity,
                metadata={
                    **chunk.metadata,
                    'chunk_id': chunk_id,
                    'entity_index': entity_index,
                    'source_file': chunk.metadata.get('file_path'),
                    'mentions': [chunk_id],
                    'node_type': 'entity'
                }
            )
            
            # Add node (deduplication handled in graph_manager)
            added = self.graph_manager.add_node(entity_node)
            if added:
                entity_node_ids.append(entity_id)
            else:
                # Entity was merged, find the existing entity ID
                existing_mentions = self.graph_manager.get_entity_mentions(entity)
                if existing_mentions:
                    entity_node_ids.append(existing_mentions[0].id)
            
        # 🔄 Create Relationship nodes (R) with dynamic numbering
        existing_relationship_count = len([node_id for node_id in self.graph_manager.graph.nodes() 
                                         if node_id.startswith(f"R_{chunk_id}_")])
        
        for i, (entity1, relation, entity2) in enumerate(extraction_result.relationships):
            # Find entity node IDs
            entity1_nodes = self.graph_manager.get_entity_mentions(entity1)
            entity2_nodes = self.graph_manager.get_entity_mentions(entity2)
            
            if entity1_nodes and entity2_nodes:
                relationship_index = existing_relationship_count + i  # Dynamic increment
                relationship_id = f"R_{chunk_id}_{relationship_index}"
                relationship_node = Node(
                    id=relationship_id,
                    type=NodeType.RELATIONSHIP,
                    content=f"{entity1} {relation} {entity2}",
                    metadata={
                        **chunk.metadata,
                        'chunk_id': chunk_id,
                        'relationship_index': relationship_index,
                        'entity1': entity1,
                        'relation': relation,
                        'entity2': entity2,
                        'source_file': chunk.metadata.get('file_path'),
                        'node_type': 'relationship'
                    }
                )
                
                self.graph_manager.add_node(relationship_node)
                
                # Create edges linking relationship to entities
                edge1 = Edge(source=relationship_id, target=entity1_nodes[0].id, relationship_type="involves_entity")
                edge2 = Edge(source=relationship_id, target=entity2_nodes[0].id, relationship_type="involves_entity")
                self.graph_manager.add_edge(edge1)
                self.graph_manager.add_edge(edge2)
                
                # Direct relationship between entities
                entity_edge = Edge(
                    source=entity1_nodes[0].id,
                    target=entity2_nodes[0].id,
                    relationship_type=relation,
                    metadata={'relationship_node': relationship_id}
                )
                self.graph_manager.add_edge(entity_edge)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk.chunk_index}: {e}")
        return False
```

---

## 🔧 Phase 2: Graph Augmentation

### Step 6: Generate Enhanced Nodes (A, H, O)
```python
# Phase 2: Choose incremental or full augmentation
if is_incremental:
    logger.info("🔄 Phase 2: Incremental Graph Augmentation")
    new_entities = pipeline.get_new_entities_from_current_file()
    augmentation_result = pipeline._phase_2_incremental_augmentation(new_entities)
else:
    logger.info("🔄 Phase 2: Full Graph Augmentation")
    augmentation_result = pipeline._phase_2_augmentation()

if not augmentation_result['success']:
    raise Exception(f"Phase 2 failed: {augmentation_result.get('error')}")
```

#### Incremental Augmentation Process:
```python
def _phase_2_incremental_augmentation(self, new_entities: List[Node]) -> Dict[str, Any]:
    """Incremental augmentation - only process new entities"""
    try:
        # 📊 Only create attribute nodes for truly new entities
        attribute_nodes_created = 0
        
        logger.info(f"Processing attributes for {len(new_entities)} new entities")
        
        for entity in new_entities:
            if self._create_attribute_node(entity):
                attribute_nodes_created += 1
                
                if attribute_nodes_created % 10 == 0:
                    logger.info(f"Created {attribute_nodes_created} attribute nodes...")
        
        # 🏘️ Community detection only for substantial new content
        communities = {}
        high_level_nodes_created = 0
        overview_nodes_created = 0
        
        if len(new_entities) > 5:  # Only if substantial new content
            logger.info("Running incremental community detection")
            communities = self.graph_manager.detect_communities()
            
            # Create community nodes for new/updated communities only
            for community_id in set(communities.values()):
                # Check if this community already has high-level nodes
                existing_high_level = any(
                    node.metadata.get('community_id') == community_id
                    for node in self.graph_manager.get_nodes_by_type(NodeType.HIGH_LEVEL)
                )
                
                if not existing_high_level:
                    high_level_created, overview_created = self._create_community_nodes(community_id)
                    if high_level_created:
                        high_level_nodes_created += 1
                    if overview_created:
                        overview_nodes_created += 1
        
        return {
            'success': True,
            'new_entities_processed': len(new_entities),
            'attribute_nodes': attribute_nodes_created,
            'communities_detected': len(communities),
            'high_level_nodes': high_level_nodes_created,
            'overview_nodes': overview_nodes_created,
            'incremental_mode': True
        }
        
    except Exception as e:
        logger.error(f"Error in Phase II incremental augmentation: {e}")
        return {'success': False, 'error': str(e)}
```

#### Attribute Node Creation:
```python
def _create_attribute_node(self, entity: Node) -> bool:
    """Create an attribute node for an important entity."""
    try:
        # Gather context from connected nodes
        connected_nodes = self.graph_manager.get_connected_nodes(entity.id)
        context_chunks = []
        
        for node in connected_nodes:
            if node.type == NodeType.TEXT:
                context_chunks.append(node.content)
        
        # Add the entity's own context
        if hasattr(entity, 'content'):
            context_chunks.append(f"Entity: {entity.content}")
        
        # Get relationships for this entity
        entity_relationships = []
        for neighbor in self.graph_manager.graph.neighbors(entity.id):
            neighbor_node = self.graph_manager.get_node(neighbor)
            if neighbor_node and neighbor_node.type == NodeType.RELATIONSHIP:
                entity_relationships.append(neighbor_node.content)
        
        # Generate attributes using LLM with relationships
        attributes = self.llm_service.generate_entity_attributes(
            entity.content, 
            context_chunks[:5],  # Limit context to avoid token limits
            entity_relationships[:3]  # Include up to 3 relationships
        )
        
        # Create attribute node with org-specific ID
        attribute_id = f"A_{entity.id}_{self.current_org_id}" if self.current_org_id else f"A_{entity.id}"
        attribute_node = Node(
            id=attribute_id,
            type=NodeType.ATTRIBUTE,
            content=attributes,
            metadata={
                'entity_id': entity.id,
                'entity_name': entity.content,
                'node_type': 'attribute'
            }
        )
        
        self.graph_manager.add_node(attribute_node)
        
        # Link to entity
        edge = Edge(
            source=attribute_id,
            target=entity.id,
            relationship_type="describes_entity"
        )
        self.graph_manager.add_edge(edge)
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating attribute node for entity {entity.id}: {e}")
        return False
```

---

## 💾 Phase 3: Embedding Generation

### Step 7: Generate Embeddings (Database-Only Mode)
```python
# Phase 3: Choose incremental or full embedding generation
if is_incremental:
    logger.info("🔄 Phase 3: Incremental Embedding Generation")
    embedding_result = pipeline._phase_3_incremental_embeddings()
else:
    logger.info("🔄 Phase 3: Full Embedding Generation")
    embedding_result = pipeline._phase_3_embedding_generation()

if not embedding_result['success']:
    raise Exception(f"Phase 3 failed: {embedding_result.get('error')}")
```

#### Full Embedding Generation Process:
```python
def _phase_3_embedding_generation(self) -> Dict[str, Any]:
    """Phase III: Generate embeddings for all nodes (Database-only storage)."""
    logger.info("Starting Phase III: Embedding Generation (Database-only mode)")
    
    # Skip HNSW for production - using database-first architecture
    logger.info("HNSW disk storage disabled for production - using database-only storage")
    
    try:
        # Get all nodes that need embeddings
        all_nodes = []
        for node_type in [NodeType.SEMANTIC, NodeType.ENTITY, NodeType.RELATIONSHIP, 
                         NodeType.ATTRIBUTE, NodeType.HIGH_LEVEL, NodeType.OVERVIEW]:
            nodes = self.graph_manager.get_nodes_by_type(node_type)
            all_nodes.extend(nodes)
        
        if not all_nodes:
            logger.warning("No nodes found for embedding generation")
            return {'success': True, 'embeddings_generated': 0}
        
        logger.info(f"Generating embeddings for {len(all_nodes)} nodes")
        
        # Prepare texts for embedding
        texts = [node.content for node in all_nodes]
        node_ids = [node.id for node in all_nodes]
        
        # Generate ALL embeddings at once with optimized batching
        logger.info(f"🚀 Generating embeddings for {len(texts)} nodes using optimized batching...")
        
        embeddings = self.llm_service.get_embeddings(texts)
        
        if len(embeddings) != len(texts):
            logger.error(f"Embedding count mismatch: {len(embeddings)} != {len(texts)}")
            return {'success': False, 'error': 'Embedding generation failed'}
        
        logger.info(f"✅ Successfully generated {len(embeddings)} embeddings")
        
        # Update nodes with embeddings (skip HNSW for production)
        embeddings_generated = 0
        
        # Update all nodes with embeddings
        for i, (node_id, embedding) in enumerate(zip(node_ids, embeddings)):
            node = self.graph_manager.get_node(node_id)
            if node and embedding is not None:
                node.embeddings = embedding
                self.graph_manager.update_node(node)
                embeddings_generated += 1
        
        logger.info(f"✅ Updated {embeddings_generated} nodes with embeddings (database-only mode)")
        
        # 🗄️ Store embeddings in PostgreSQL database
        try:
            if embeddings_generated > 0:
                logger.info(f"Storing {embeddings_generated} embeddings in database")
                
                # Prepare embedding data for bulk storage
                embedding_data = []
                for i, (node, embedding) in enumerate(zip(all_nodes, embeddings)):
                    if embedding is not None and len(embedding) > 0:
                        embedding_data.append({
                            'node_id': node.id,
                            'node_type': node.type.value,
                            'content': node.content,
                            'embedding': embedding,  # Use actual embedding from array
                            'metadata': node.metadata,
                            'chunk_index': node.metadata.get('chunk_index', 0)
                        })
                
                logger.info(f"📊 Prepared {len(embedding_data)} embedding records for storage")
                
                if embedding_data:
                    # Get storage service and store embeddings
                    from src.storage.neon_storage import NeonDBStorage
                    storage = NeonDBStorage()
                    
                    # Get user_id from node metadata
                    user_id = 'unknown'
                    if embedding_data and embedding_data[0].get('metadata'):
                        user_id = embedding_data[0]['metadata'].get('user_id', 'unknown')
                    
                    logger.info(f"🗄️ Calling bulk_store_embeddings with {len(embedding_data)} items...")
                    
                    storage_result = storage.bulk_store_embeddings(
                        org_id=self.current_org_id,
                        file_id=self.current_file_id,
                        user_id=user_id,
                        embedding_data=embedding_data
                    )
                    
                    logger.info(f"📤 Storage result: {storage_result}")
                    
                    if storage_result.get('success'):
                        logger.info(f"✅ Stored {storage_result.get('stored_count', 0)} embeddings in database")
                    else:
                        logger.error(f"❌ Failed to store embeddings in database: {storage_result.get('error')}")
                        
        except Exception as e:
            logger.warning(f"Failed to store embeddings in database: {e}")
        
        logger.info(f"Phase III completed: {embeddings_generated} embeddings generated and indexed")
        
        return {
            'success': True,
            'embeddings_generated': embeddings_generated,
            'database_stored': embeddings_generated,
            'storage_mode': 'database_only'
        }
        
    except Exception as e:
        logger.error(f"Error in Phase III embedding generation: {e}")
        return {
            'success': False,
            'error': str(e),
            'embeddings_generated': 0
        }
```

### Step 8: Store Unified Org Graph
```python
# 🗄️ STEP 4: Store updated org graph
logger.info("💾 Storing org-level graph in NeonDB")

# Get updated processed files list
updated_processed_files = existing_graph.get('processed_files', []) if existing_graph else []
if file_id not in updated_processed_files:
    updated_processed_files.append(file_id)

# Get org stats
org_stats = pipeline.graph_manager.get_org_stats(updated_processed_files)

# Store unified org graph
import pickle
graph_data = pickle.dumps({
    'graph': pipeline.graph_manager.graph,
    'entity_index': dict(pipeline.graph_manager.entity_index),
    'community_assignments': pipeline.graph_manager.community_assignments
})

version = (existing_graph.get('version', 0) + 1) if existing_graph else 1

storage_result = storage.store_org_graph_sync(
    org_id=org_id,
    graph_data=graph_data,
    processed_files=updated_processed_files,
    version=version,
    last_file_added=file_id,
    stats=org_stats,
    user_id=user_id
)

if not storage_result.get("success"):
    raise Exception(f"Org graph storage failed: {storage_result.get('error')}")

logger.info(f"✅ Org-level processing completed for file_id={file_id}, org_id={org_id} (v{version})")
```

### Response Format
```json
{
    "message": "Document processing started",
    "file_id": "file-uuid",
    "status": "processing",
    "estimated_time": "2-5 minutes"
}
```

---

## 🔍 2. GENERATE-RESPONSE FLOW

### API Endpoint
```
POST /api/v1/generate-response
```

### Request Format
```json
{
    "query": "What is machine learning?",
    "org_id": "organization-uuid",
    "max_results": 5,
    "max_tokens": 2048,
    "temperature": 0.7,
    "conversation_history": ""
}
```

### Step 1: Request Validation & Graph Loading
```python
@app.route("/api/v1/generate-response", methods=["POST"])
def generate_response():
    """Generate a response using NodeRAG's advanced search system"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if "query" not in data or "org_id" not in data:
            return jsonify({"error": "Missing required fields: query, org_id"}), 400
        
        query = data["query"]
        org_id = data["org_id"]
        max_results = data.get("max_results", 5)
        
        logger.info(f"🔍 NodeRAG query request: org_id={org_id}, query='{query[:100]}...'")
        
        # 🔍 Check if org has any data by loading org graph first
        storage = noderag_service.get_neon_storage()
        existing_graph = storage.load_org_graph_sync(org_id)
        
        if not existing_graph:
            logger.warning(f"No graph data found for org {org_id}")
            return jsonify({
                "error": "No knowledge base found for organization",
                "message": f"No processed documents found for org {org_id}. Please process some documents first.",
                "org_id": org_id
            }), 404
        
        # Load the org graph into the pipeline
        pipeline = noderag_service.get_pipeline()
        logger.info(f"Loading org graph for {org_id} into search system")
        pipeline.graph_manager.load_from_data(existing_graph['graph_data'])
        
        # Reset and get the advanced search system with loaded graph
        noderag_service.advanced_search = None  # Reset to force reinitialization
        advanced_search = noderag_service.get_advanced_search()
        
        if not advanced_search:
            logger.error("Failed to initialize AdvancedSearchSystem after loading graph")
            return jsonify({"error": "Failed to initialize search system"}), 500
        
        # Set org_id for database vector search
        advanced_search._current_org_id = org_id
        
        # Perform the search and generate response
        search_result = advanced_search.answer_query(
            query=query,
            use_structured_prompt=True
        )
        
        if search_result.get('error'):
            logger.error(f"NodeRAG search failed: {search_result.get('answer', 'Unknown error')}")
            return jsonify({
                "error": "Search failed",
                "message": search_result.get('answer', 'Unknown error')
            }), 500
        
        logger.info(f"✅ NodeRAG response generated successfully for org {org_id}")
        
        return jsonify({
            "success": True,
            "response": search_result['answer'],
            "context": {
                "retrieved_nodes": search_result.get('retrieved_nodes', 0),
                "context_length": search_result.get('context_length', 0),
                "retrieval_metadata": search_result.get('retrieval_metadata', {}),
                "org_id": org_id,
                "query": query
            },
            "metadata": search_result.get('retrieval_metadata', {})
        })
        
    except Exception as e:
        logger.error(f"❌ Generate response error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500
```

---

## 🎯 Phase 1: Multi-Signal Search

### Step 2: Initialize Advanced Search System
```python
# Reset and get the advanced search system with loaded graph
noderag_service.advanced_search = None  # Force reinitialization
advanced_search = noderag_service.get_advanced_search()

# Set org_id for database vector search
advanced_search._current_org_id = org_id
```

### Step 3: Perform Multi-Signal Retrieval
```python
search_result = advanced_search.answer_query(query=query, use_structured_prompt=True)

def answer_query(self, query: str, use_structured_prompt: bool = True):
    # 🔍 Perform comprehensive search
    retrieval_result = self.search(query)
    
def search(self, query: str):
    # 🚀 Step 1: Database Vector Similarity Search (replaces HNSW)
    hnsw_results = self._hnsw_search(query, k_hnsw)
    
    # 🎯 Step 2: Exact Entity Matching
    accurate_results = self._exact_entity_search(query)
    
    # 📊 Step 3: Personalized PageRank
    ppr_results = self._personalized_pagerank_search(hnsw_results, accurate_results, k_final)
    
    # 🏗️ Step 4: Post-process and Categorize Results
    final_result = self._post_process_results(...)
    
    return final_result
```

### Step 4: Database Vector Search
```python
def _hnsw_search(self, query: str, k: int) -> List[SearchResult]:
    """Perform database vector similarity search (replaces HNSW)."""
    try:
        # Generate query embedding
        query_embeddings = self.llm_service.get_embeddings([query])
        if not query_embeddings or not query_embeddings[0]:
            logger.warning("Failed to get query embedding")
            return []
        
        query_embedding = query_embeddings[0]
        
        # Get current org_id from search context
        org_id = getattr(self, '_current_org_id', None)
        if not org_id:
            # Try to get org_id from first node's metadata
            for node_id, node_data in self.graph_manager.graph.nodes(data=True):
                metadata = node_data.get('metadata', {})
                if 'org_id' in metadata:
                    org_id = metadata['org_id']
                    self._current_org_id = org_id
                    break
            
            if not org_id:
                logger.warning("No org_id found for database vector search")
                return []
        
        # 🗄️ Perform database vector search using PostgreSQL pgvector
        from src.storage.neon_storage import NeonDBStorage
        storage = NeonDBStorage()
        
        search_results = storage.vector_similarity_search_sync(
            query_embedding=query_embedding,
            org_id=org_id,
            k=k,
            similarity_threshold=0.5  # Lower threshold for more results
        )
        
        # Convert database results to SearchResult format
        results = []
        for result in search_results:
            # Create SearchResult-like object with both similarity and distance
            search_result = type('SearchResult', (), {
                'node_id': result['node_id'],
                'similarity': result['similarity_score'],
                'distance': 1.0 - result['similarity_score'],  # Convert similarity to distance
                'content': result['content'],
                'node_type': result['node_type'],
                'metadata': result['metadata']
            })()
            results.append(search_result)
        
        logger.debug(f"Database vector search returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error in database vector search: {e}")
        return []
```

### Step 5: Exact Entity Search
```python
def _exact_entity_search(self, query: str) -> List[str]:
    """Perform exact entity matching using query decomposition."""
    try:
        # 🔍 Decompose query into entities using LLM
        decomposed_entities = self.llm_service.decompose_query(query)
        
        matched_entities = []
        
        # Search for exact matches
        for entity in decomposed_entities:
            entity_lower = entity.lower().strip()
            
            # Direct lookup in entity index
            if entity_lower in self.entity_lookup:
                matched_entities.extend(self.entity_lookup[entity_lower])
                continue
            
            # Fuzzy word-based matching
            entity_words = entity_lower.split()
            if len(entity_words) > 1:
                # Create regex pattern for phrase matching
                pattern = re.compile(r'\b' + r'\s+'.join(map(re.escape, entity_words)) + r'\b')
                
                for entity_text, node_ids in self.entity_lookup.items():
                    if pattern.search(entity_text):
                        matched_entities.extend(node_ids)
        
        # Remove duplicates while preserving order
        unique_entities = []
        seen = set()
        for entity_id in matched_entities:
            if entity_id not in seen:
                unique_entities.append(entity_id)
                seen.add(entity_id)
        
        logger.debug(f"Exact entity search found {len(unique_entities)} matches")
        return unique_entities
        
    except Exception as e:
        logger.error(f"Error in exact entity search: {e}")
        return []
```

### Step 6: Personalized PageRank
```python
def _personalized_pagerank_search(self, 
                                hnsw_results: List[SearchResult],
                                accurate_results: List[str],
                                k: int) -> List[Tuple[str, float]]:
    """Perform personalized PageRank search."""
    try:
        # 🎯 Build personalization vector
        personalization = {}
        
        # Add HNSW results with similarity weight
        for result in hnsw_results:
            # Convert distance to similarity (assuming cosine distance)
            similarity = 1.0 - result.distance
            personalization[result.node_id] = similarity * Config.SIMILARITY_WEIGHT
        
        # Add exact matches with accuracy weight
        for node_id in accurate_results:
            personalization[node_id] = Config.ACCURACY_WEIGHT
        
        if not personalization:
            logger.warning("No seed nodes for PPR")
            return []
        
        # 📊 Compute Personalized PageRank
        ppr_results = self.ppr.search_top_k(
            personalization=personalization,
            k=k * 2,  # Get more results for filtering
            alpha=Config.PPR_ALPHA,
            max_iter=Config.PPR_MAX_ITERATIONS
        )
        
        logger.debug(f"PPR search returned {len(ppr_results)} results")
        return ppr_results
        
    except Exception as e:
        logger.error(f"Error in PPR search: {e}")
        return []
```

### Step 7: Result Post-Processing & Categorization
```python
def _post_process_results(self,
                        query: str,
                        hnsw_results: List[SearchResult],
                        accurate_results: List[str],
                        ppr_results: List[Tuple[str, float]],
                        entity_limit: int,
                        relationship_limit: int,
                        high_level_limit: int) -> RetrievalResult:
    """Post-process and categorize search results."""
    
    # Track already selected nodes
    selected_nodes = set()
    
    # Add HNSW and exact match results first
    for result in hnsw_results:
        selected_nodes.add(result.node_id)
    
    for node_id in accurate_results:
        selected_nodes.add(node_id)
    
    # 📂 Categorize nodes by type
    entity_nodes = []
    relationship_nodes = []
    high_level_nodes = []
    other_nodes = []
    
    # Process PPR results and categorize
    for node_id, score in ppr_results:
        if node_id in selected_nodes:
            continue
        
        node = self.graph_manager.get_node(node_id)
        if not node:
            continue
        
        # Categorize by node type
        if node.type == NodeType.ENTITY:
            if len(entity_nodes) < entity_limit:
                entity_nodes.append(node_id)
                
        elif node.type == NodeType.RELATIONSHIP:
            if len(relationship_nodes) < relationship_limit:
                relationship_nodes.append(node_id)
                
        elif node.type in [NodeType.HIGH_LEVEL, NodeType.OVERVIEW]:
            if len(high_level_nodes) < high_level_limit:
                high_level_nodes.append(node_id)
                
        else:
            other_nodes.append(node_id)
        
        selected_nodes.add(node_id)
        
        # Stop if we have enough in all categories
        if (len(entity_nodes) >= entity_limit and 
            len(relationship_nodes) >= relationship_limit and 
            len(high_level_nodes) >= high_level_limit):
            break
    
    # 🔗 Add entity attributes for selected entities
    final_nodes = list(selected_nodes)
    for entity_id in entity_nodes:
        # Find attribute nodes for this entity
        connected_nodes = self.graph_manager.get_connected_nodes(entity_id)
        for connected in connected_nodes:
            if (connected.type == NodeType.ATTRIBUTE and 
                connected.id not in selected_nodes):
                final_nodes.append(connected.id)
                selected_nodes.add(connected.id)
    
    # Add high-level content for high-level titles
    for hl_title_id in high_level_nodes:
        connected_nodes = self.graph_manager.get_connected_nodes(hl_title_id)
        for connected in connected_nodes:
            if (connected.type == NodeType.HIGH_LEVEL and 
                connected.id not in selected_nodes):
                final_nodes.append(connected.id)
                selected_nodes.add(connected.id)
                break  # Only add one high-level content per title
    
    # Build metadata
    search_metadata = {
        'hnsw_count': len(hnsw_results),
        'exact_match_count': len(accurate_results),
        'ppr_count': len(ppr_results),
        'entity_count': len(entity_nodes),
        'relationship_count': len(relationship_nodes),
        'high_level_count': len(high_level_nodes),
        'total_selected': len(final_nodes)
    }
    
    return RetrievalResult(
        query=query,
        hnsw_results=hnsw_results,
        accurate_results=accurate_results,
        ppr_results=ppr_results,
        final_nodes=final_nodes,
        entity_nodes=entity_nodes,
        relationship_nodes=relationship_nodes,
        high_level_nodes=high_level_nodes,
        search_metadata=search_metadata
    )
```

---

## 🤖 Phase 2: Answer Generation

### Step 8: Context Building & LLM Response
```python
def answer_query(self, query: str, use_structured_prompt: bool = True):
    """Generate answer for query using retrieved information."""
    try:
        # Perform search
        retrieval_result = self.search(query)
        
        # 🏗️ Build context from retrieved nodes
        context_parts = []
        
        for node_id in retrieval_result.final_nodes:
            node = self.graph_manager.get_node(node_id)
            if node:
                if use_structured_prompt:
                    context_parts.append(f"[{node.type.value}] {node.content}")
                else:
                    context_parts.append(node.content)
        
        retrieved_info = "\n\n".join(context_parts)
        
        # 🤖 Generate answer using LLM
        answer_prompt = self.llm_service.prompt_manager.answer_generation.format(
            info=retrieved_info,
            query=query
        )
        
        response = self.llm_service._chat_completion(answer_prompt, temperature=0.7)
        
        return {
            'query': query,
            'answer': response,
            'retrieval_metadata': retrieval_result.search_metadata,
            'retrieved_nodes': len(retrieval_result.final_nodes),
            'context_length': len(retrieved_info)
        }
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            'query': query,
            'answer': f"Error generating answer: {str(e)}",
            'error': True
        }
```

### Step 9: Final Response
```python
return jsonify({
    "success": True,
    "response": search_result['answer'],
    "context": {
        "retrieved_nodes": search_result.get('retrieved_nodes', 0),
        "context_length": search_result.get('context_length', 0),
        "retrieval_metadata": search_result.get('retrieval_metadata', {}),
        "org_id": org_id,
        "query": query
    },
    "metadata": search_result.get('retrieval_metadata', {})
})
```

### Response Format
```json
{
    "success": true,
    "response": "Machine learning is a subset of artificial intelligence...",
    "context": {
        "retrieved_nodes": 15,
        "context_length": 2048,
        "retrieval_metadata": {
            "hnsw_count": 5,
            "exact_match_count": 3,
            "ppr_count": 20,
            "entity_count": 8,
            "relationship_count": 4,
            "high_level_count": 3,
            "total_selected": 15
        },
        "org_id": "org-uuid",
        "query": "What is machine learning?"
    }
}
```

---

## 🗑️ 3. DELETE-EMBEDDINGS FLOW

### API Endpoint
```
DELETE /api/v1/delete-embeddings
```

### Request Format
```json
{
    "org_id": "organization-uuid",
    "file_id": "file-uuid"  // Optional - if not provided, deletes all org data
}
```

### Implementation
```python
@app.route("/api/v1/delete-embeddings", methods=["DELETE"])
def delete_embeddings():
    """Delete embeddings for an organization or specific file"""
    try:
        data = request.get_json()
        
        if "org_id" not in data:
            return jsonify({"error": "Missing required field: org_id"}), 400
        
        org_id = data["org_id"]
        file_id = data.get("file_id")  # Optional
        
        logger.info(f"🗑️ Delete request: org_id={org_id}, file_id={file_id}")
        
        storage = noderag_service.get_neon_storage()
        
        if file_id:
            # Delete specific file embeddings
            result = storage.delete_file_data(org_id, file_id)
            
            if result.get("success"):
                return jsonify({
                    "success": True,
                    "message": f"Deleted embeddings for file {file_id}",
                    "org_id": org_id,
                    "file_id": file_id,
                    "embeddings_deleted": result.get("deleted_count", 0),
                    "graphs_deleted": result.get("graphs_deleted", 0)
                })
            else:
                return jsonify({
                    "error": "Failed to delete file embeddings",
                    "message": result.get("error", "Unknown error")
                }), 500
        else:
            # Full org deletion not implemented yet
            return jsonify({
                "error": "Full organization deletion not implemented",
                "message": "Please provide file_id parameter to delete specific file"
            }), 400
        
    except Exception as e:
        logger.error(f"❌ Delete embeddings error: {e}")
        return jsonify({"error": "Internal server error", "message": str(e)}), 500
```

---

## 📊 4. STATUS CHECK FLOW

### API Endpoint
```
GET /api/v1/status/{file_id}
```

### Implementation
```python
@app.route("/api/v1/status/<file_id>", methods=["GET"])
def get_status(file_id: str):
    """Get processing status for a file"""
    try:
        with processing_lock:
            status = processing_status.get(file_id)
        
        if not status:
            return jsonify({"error": "File not found"}), 404
        
        return jsonify({
            "file_id": file_id,
            **status
        })
        
    except Exception as e:
        logger.error(f"❌ Status check error: {e}")
        return jsonify({"error": str(e)}), 500
```

### Response Format
```json
{
    "file_id": "file-uuid",
    "status": "completed",
    "phase": "completed", 
    "progress": 100,
    "started_at": 1634567890.123,
    "completed_at": 1634567950.456,
    "results": {
        "chunks_processed": 25,
        "incremental_mode": true,
        "org_version": 3,
        "org_total_files": 5,
        "org_graph_nodes": 1250,
        "org_graph_edges": 2100
    }
}
```

---

## 🔄 Key Features Summary

### 🏗️ **Architecture Features**
1. **Org-Level Management**: Single unified graph per organization
2. **Incremental Processing**: New files enhance existing graphs without rebuilding
3. **Dynamic Section Numbering**: Automatically increments based on existing nodes
4. **Concurrent Protection**: Org-level locks prevent processing conflicts
5. **Database-First Storage**: All embeddings stored in PostgreSQL with pgvector

### 🚀 **Performance Features**
1. **Production Optimized**: No HNSW disk storage, prevents storage bloat
2. **Optimized Batching**: Processes embeddings in efficient batches
3. **CPU Optimization**: Handles single CPU environments with controlled batching
4. **Memory Efficient**: Uses lazy initialization for services

### 🔍 **Search Features**
1. **Multi-Signal Retrieval**: Combines vector search, exact matching, and PageRank
2. **Structured Results**: Categorizes nodes by type (entities, relationships, high-level)
3. **Context-Aware**: Includes entity attributes and related content
4. **Similarity Scoring**: Provides relevance scoring for results

### 📊 **Monitoring Features**
1. **Progress Tracking**: Real-time status updates during processing
2. **Webhook Support**: Notifications for processing completion/failure
3. **Detailed Metrics**: Comprehensive statistics on graph structure
4. **Error Handling**: Robust error handling with detailed logging

This complete flow enables organizations to build and query their knowledge graphs through a simple REST API interface with enterprise-grade performance and reliability.