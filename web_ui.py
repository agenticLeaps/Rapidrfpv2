import streamlit as st
import requests
import json
import os
import time
import traceback
from typing import List, Dict, Any
import tempfile
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="RapidRFP RAG System",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_stats" not in st.session_state:
    st.session_state.graph_stats = {}
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:5001"

class RapidRFPAPI:
    """API client for RapidRFP RAG system."""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url.rstrip('/')
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None, files: Dict = None) -> Dict:
        """Make HTTP request to API."""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == 'GET':
                response = requests.get(url, timeout=300)
            elif method.upper() == 'POST':
                if files:
                    response = requests.post(url, data=data, files=files, timeout=300)
                else:
                    response = requests.post(url, json=data, timeout=300)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'error': f'HTTP {response.status_code}',
                    'details': response.text
                }
        except Exception as e:
            return {
                'error': str(e),
                'details': traceback.format_exc()
            }
    
    def health_check(self) -> Dict:
        """Check API health."""
        return self._make_request('GET', '/health')
    
    def index_document(self, file_path: str) -> Dict:
        """Index a document."""
        return self._make_request('POST', '/api/index/document', {'file_path': file_path})
    
    def upload_document(self, file) -> Dict:
        """Upload and index a document."""
        return self._make_request('POST', '/api/upload/document', files={'file': file})
    
    def get_graph_stats(self) -> Dict:
        """Get graph statistics."""
        return self._make_request('GET', '/api/graph/stats')
    
    def get_nodes_by_type(self, node_type: str) -> Dict:
        """Get nodes by type."""
        return self._make_request('GET', f'/api/graph/nodes/{node_type}')
    
    def search_entities(self, entity_name: str) -> Dict:
        """Search for entities."""
        return self._make_request('POST', '/api/search/entities', {'entity_name': entity_name})
    
    def advanced_search(self, query: str, top_k: int = 10) -> Dict:
        """Perform advanced search."""
        return self._make_request('POST', '/api/search/advanced', {'query': query, 'top_k': top_k})
    
    def vector_search(self, query: str, k: int = 10) -> Dict:
        """Perform vector search."""
        return self._make_request('POST', '/api/search/vector', {'query': query, 'k': k})
    
    def ppr_search(self, seed_nodes: List[str], alpha: float = 0.85, top_k: int = 20) -> Dict:
        """Perform Personalized PageRank search."""
        return self._make_request('POST', '/api/search/ppr', {
            'seed_nodes': seed_nodes, 
            'alpha': alpha, 
            'top_k': top_k
        })
    
    def get_communities(self) -> Dict:
        """Get graph communities."""
        return self._make_request('GET', '/api/graph/communities')
    
    def get_important_entities(self) -> Dict:
        """Get important entities."""
        return self._make_request('GET', '/api/graph/important-entities')
    
    def create_visualization(self, max_nodes: int = 2000, filter_node_types: List[str] = None, 
                           highlight_communities: bool = False, output_filename: str = None) -> Dict:
        """Create graph visualization."""
        data = {
            'max_nodes': max_nodes,
            'highlight_communities': highlight_communities
        }
        if filter_node_types:
            data['filter_node_types'] = filter_node_types
        if output_filename:
            data['output_filename'] = output_filename
        
        return self._make_request('POST', '/api/visualization/create', data)
    
    def create_community_visualization(self, output_filename: str = None) -> Dict:
        """Create community visualization."""
        data = {}
        if output_filename:
            data['output_filename'] = output_filename
        return self._make_request('POST', '/api/visualization/community', data)
    
    def create_node_type_visualization(self, node_types: List[str], output_filename: str = None) -> Dict:
        """Create node type visualization."""
        data = {'node_types': node_types}
        if output_filename:
            data['output_filename'] = output_filename
        return self._make_request('POST', '/api/visualization/node-types', data)
    
    def get_visualization_stats(self) -> Dict:
        """Get visualization statistics."""
        return self._make_request('GET', '/api/visualization/stats')
    
    def answer_query(self, query: str, use_structured_prompt: bool = True, 
                    k_hnsw: int = 10, k_final: int = 20,
                    entity_limit: int = 10, relationship_limit: int = 30, 
                    high_level_limit: int = 10) -> Dict:
        """Generate comprehensive answer using advanced search and retrieval."""
        return self._make_request('POST', '/api/answer', {
            'query': query,
            'use_structured_prompt': use_structured_prompt,
            'k_hnsw': k_hnsw,
            'k_final': k_final,
            'entity_limit': entity_limit,
            'relationship_limit': relationship_limit,
            'high_level_limit': high_level_limit
        })

def display_header():
    """Display the main header."""
    st.title("🔗 RapidRFP RAG System")
    st.markdown("""
    **Graph-based Retrieval Augmented Generation** system with advanced document processing, 
    knowledge graph construction, and multi-signal search capabilities.
    """)
    
    # API Status
    api = RapidRFPAPI(st.session_state.api_base_url)
    health = api.health_check()
    
    if 'error' not in health:
        st.success(f"✅ API Status: {health.get('status', 'Unknown')}")
    else:
        st.error(f"❌ API Error: {health.get('error', 'Unknown')}")

def display_sidebar():
    """Display the sidebar with controls."""
    with st.sidebar:
        st.title("📊 System Controls")
        
        # API Configuration
        with st.expander("🔧 API Configuration", expanded=False):
            new_url = st.text_input(
                "API Base URL", 
                value=st.session_state.api_base_url,
                help="Base URL for the RapidRFP RAG API"
            )
            if new_url != st.session_state.api_base_url:
                st.session_state.api_base_url = new_url
                st.rerun()
        
        # Document Upload
        with st.expander("📁 Document Upload", expanded=True):
            uploaded_file = st.file_uploader(
                "Choose a document",
                type=['pdf', 'docx', 'txt', 'md'],
                help="Upload a document to be processed"
            )
            
            if uploaded_file is not None:
                if st.button("🚀 Process Document", type="primary"):
                    with st.spinner("Processing document..."):
                        api = RapidRFPAPI(st.session_state.api_base_url)
                        result = api.upload_document(uploaded_file)
                        
                        if 'error' not in result:
                            st.success("Document processed successfully!")
                            st.json(result)
                            # Refresh graph stats
                            st.session_state.graph_stats = api.get_graph_stats()
                        else:
                            st.error(f"Error: {result['error']}")
        
        # File Path Processing
        with st.expander("📂 File Path Processing", expanded=False):
            file_path = st.text_input("Document File Path")
            
            if file_path and st.button("🚀 Process File Path"):
                if os.path.exists(file_path):
                    with st.spinner("Processing document..."):
                        api = RapidRFPAPI(st.session_state.api_base_url)
                        result = api.index_document(file_path)
                        
                        if 'error' not in result:
                            st.success("Document processed successfully!")
                            st.json(result)
                            st.session_state.graph_stats = api.get_graph_stats()
                        else:
                            st.error(f"Error: {result['error']}")
                else:
                    st.error("File not found!")
        
        # Graph Statistics
        with st.expander("📈 Graph Statistics", expanded=True):
            if st.button("🔄 Refresh Stats"):
                api = RapidRFPAPI(st.session_state.api_base_url)
                st.session_state.graph_stats = api.get_graph_stats()
            
            if st.session_state.graph_stats and 'error' not in st.session_state.graph_stats:
                stats = st.session_state.graph_stats
                st.metric("Total Nodes", stats.get('total_nodes', 0))
                st.metric("Total Edges", stats.get('total_edges', 0))
                st.metric("Communities", stats.get('num_communities', 0))
                
                # Node type breakdown
                node_counts = stats.get('node_type_counts', {})
                if node_counts:
                    st.markdown("**Node Types:**")
                    for node_type, count in node_counts.items():
                        st.write(f"• {node_type}: {count}")

def display_search_interface():
    """Display the search interface."""
    st.header("🔍 Search & Query Interface")
    
    # Search tabs
    search_tab, entity_tab, advanced_tab, answer_tab = st.tabs(["Basic Search", "Entity Search", "Advanced Search", "Answer Generation"])
    
    with search_tab:
        st.subheader("Multi-Signal Search")
        query = st.text_input("Enter your search query:", key="basic_search")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            top_k = st.slider("Results", 1, 50, 10, key="basic_top_k")
        
        if st.button("🔍 Search", key="basic_search_btn") and query:
            with st.spinner("Searching..."):
                api = RapidRFPAPI(st.session_state.api_base_url)
                results = api.advanced_search(query, top_k)
                
                if 'error' not in results:
                    st.success(f"Found {results.get('total_results', 0)} results")
                    display_search_results(results.get('results', []))
                else:
                    st.error(f"Search error: {results['error']}")
    
    with entity_tab:
        st.subheader("Entity Search")
        entity_name = st.text_input("Enter entity name:", key="entity_search")
        
        if st.button("🔍 Search Entity", key="entity_search_btn") and entity_name:
            with st.spinner("Searching entities..."):
                api = RapidRFPAPI(st.session_state.api_base_url)
                results = api.search_entities(entity_name)
                
                if 'error' not in results:
                    st.success(f"Found {results.get('count', 0)} entities")
                    entities = results.get('entities', [])
                    for entity in entities:
                        with st.expander(f"Entity: {entity.get('content', 'Unknown')}"):
                            st.json(entity)
                else:
                    st.error(f"Search error: {results['error']}")
    
    with advanced_tab:
        st.subheader("Advanced Search Options")
        
        # Vector Search
        st.markdown("**Vector Similarity Search**")
        vector_query = st.text_input("Vector Search Query:", key="vector_query")
        vector_k = st.slider("Vector Results", 1, 50, 10, key="vector_k")
        
        if st.button("🔍 Vector Search", key="vector_search_btn") and vector_query:
            with st.spinner("Performing vector search..."):
                api = RapidRFPAPI(st.session_state.api_base_url)
                results = api.vector_search(vector_query, vector_k)
                
                if 'error' not in results:
                    st.success(f"Found {results.get('total_results', 0)} results")
                    display_vector_results(results.get('results', []))
                else:
                    st.error(f"Vector search error: {results['error']}")
        
        # PPR Search
        st.markdown("**Personalized PageRank Search**")
        seed_nodes_input = st.text_area("Seed Nodes (one per line):", key="ppr_seeds")
        ppr_alpha = st.slider("PPR Alpha", 0.0, 1.0, 0.85, 0.05, key="ppr_alpha")
        ppr_k = st.slider("PPR Results", 1, 50, 20, key="ppr_k")
        
        if st.button("🔍 PPR Search", key="ppr_search_btn") and seed_nodes_input:
            seed_nodes = [line.strip() for line in seed_nodes_input.split('\n') if line.strip()]
            
            with st.spinner("Performing PageRank search..."):
                api = RapidRFPAPI(st.session_state.api_base_url)
                results = api.ppr_search(seed_nodes, ppr_alpha, ppr_k)
                
                if 'error' not in results:
                    st.success(f"Found {results.get('total_results', 0)} results")
                    display_ppr_results(results.get('results', []))
                else:
                    st.error(f"PPR search error: {results['error']}")
    
    with answer_tab:
        st.subheader("🤖 AI Answer Generation")
        st.markdown("""
        **Comprehensive Q&A System** using advanced retrieval and graph-based reasoning.
        Get detailed answers backed by retrieved knowledge.
        """)
        
        # Query input
        answer_query = st.text_area("Enter your question:", 
                                   placeholder="Ask a detailed question about your documents...",
                                   key="answer_query", height=100)
        
        # Advanced options
        with st.expander("🔧 Advanced Answer Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                use_structured = st.checkbox("Use structured prompts", value=True, 
                                           help="Group retrieved content by type for better reasoning")
                k_hnsw = st.slider("HNSW Results", 5, 30, 10, 
                                  help="Number of semantically similar results")
                k_final = st.slider("Final Results", 10, 50, 20,
                                   help="Total number of nodes for reasoning")
            
            with col2:
                entity_limit = st.slider("Entity Nodes", 5, 20, 10,
                                       help="Max entity nodes to include")
                relationship_limit = st.slider("Relationship Nodes", 10, 50, 30,
                                             help="Max relationship nodes to include")
                high_level_limit = st.slider("High-level Nodes", 5, 20, 10,
                                           help="Max high-level concept nodes")
        
        # Generate answer
        if st.button("🤖 Generate Answer", key="generate_answer_btn", type="primary") and answer_query:
            with st.spinner("🔍 Searching knowledge graph and generating answer..."):
                api = RapidRFPAPI(st.session_state.api_base_url)
                
                try:
                    result = api.answer_query(
                        query=answer_query,
                        use_structured_prompt=use_structured,
                        k_hnsw=k_hnsw,
                        k_final=k_final,
                        entity_limit=entity_limit,
                        relationship_limit=relationship_limit,
                        high_level_limit=high_level_limit
                    )
                    
                    if result.get('success'):
                        # Display answer
                        st.markdown("### 🎯 Answer")
                        answer_container = st.container()
                        with answer_container:
                            st.markdown(f"**Query:** {result['query']}")
                            st.markdown("**Response:**")
                            st.info(result['answer'])
                        
                        # Display retrieval metadata
                        st.markdown("### 📊 Retrieval Details")
                        metadata = result.get('retrieval_metadata', {})
                        breakdown = result.get('search_breakdown', {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Nodes", metadata.get('total_nodes_retrieved', 0))
                            st.metric("HNSW Results", metadata.get('hnsw_results', 0))
                        with col2:
                            st.metric("Exact Matches", metadata.get('exact_matches', 0))
                            st.metric("PPR Results", metadata.get('ppr_results', 0))
                        with col3:
                            st.metric("Context Length", metadata.get('context_length', 0))
                            st.metric("Structured", "Yes" if metadata.get('use_structured_prompt') else "No")
                        
                        # Node type breakdown
                        st.markdown("**Retrieved Node Types:**")
                        breakdown_cols = st.columns(5)
                        breakdown_labels = ["Entities", "Relationships", "High-level", "Attributes", "Other"]
                        breakdown_keys = ["entity_count", "relationship_count", "high_level_count", "attribute_count", "other_count"]
                        
                        for col, label, key in zip(breakdown_cols, breakdown_labels, breakdown_keys):
                            col.metric(label, breakdown.get(key, 0))
                        
                        # Show retrieved nodes
                        if st.checkbox("Show Retrieved Nodes", key="show_nodes"):
                            st.markdown("### 📋 Retrieved Knowledge Nodes")
                            nodes = result.get('retrieved_nodes', [])
                            
                            for i, node in enumerate(nodes):
                                with st.expander(f"{node['type']} - {node['id'][:30]}..."):
                                    st.write("**Type:**", node['type'])
                                    st.write("**Content:**")
                                    st.text(node['content'])
                    
                    else:
                        st.error(f"Answer generation failed: {result.get('error', 'Unknown error')}")
                        if 'traceback' in result:
                            with st.expander("Error Details"):
                                st.code(result['traceback'])
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

def display_search_results(results: List[Dict]):
    """Display search results."""
    if not results:
        st.info("No results found.")
        return
    
    for i, result in enumerate(results):
        with st.expander(f"Result {i+1}: {result.get('type', 'Unknown')} - {result.get('id', 'Unknown')[:20]}..."):
            st.write("**Content:**")
            st.write(result.get('content', 'No content'))
            
            st.write("**Metadata:**")
            metadata = result.get('metadata', {})
            for key, value in metadata.items():
                st.write(f"• {key}: {value}")
            
            if 'score' in result:
                st.metric("Relevance Score", f"{result['score']:.4f}")

def display_vector_results(results: List[Dict]):
    """Display vector search results."""
    if not results:
        st.info("No results found.")
        return
    
    for i, result in enumerate(results):
        with st.expander(f"Result {i+1}: {result.get('id', 'Unknown')[:20]}..."):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Similarity", f"{result.get('similarity', 0):.4f}")
            with col2:
                st.metric("Distance", f"{result.get('distance', 0):.4f}")

def display_ppr_results(results: List[Dict]):
    """Display PPR search results."""
    if not results:
        st.info("No results found.")
        return
    
    for i, result in enumerate(results):
        with st.expander(f"Result {i+1}: {result.get('node_id', 'Unknown')[:20]}..."):
            st.metric("PageRank Score", f"{result.get('score', 0):.6f}")
            if 'node_info' in result:
                st.json(result['node_info'])

def display_graph_analysis():
    """Display graph analysis tools."""
    st.header("📊 Graph Analysis")
    
    analysis_tab, community_tab, entities_tab = st.tabs(["Node Analysis", "Communities", "Important Entities"])
    
    with analysis_tab:
        st.subheader("Node Type Analysis")
        
        # Node type selector
        node_types = ["T", "S", "N", "R", "A", "H", "O"]
        selected_type = st.selectbox("Select Node Type:", node_types)
        
        if st.button("📋 Get Nodes", key="get_nodes_btn"):
            with st.spinner(f"Fetching {selected_type} nodes..."):
                api = RapidRFPAPI(st.session_state.api_base_url)
                results = api.get_nodes_by_type(selected_type)
                
                if 'error' not in results:
                    nodes = results.get('nodes', [])
                    st.success(f"Found {len(nodes)} nodes of type {selected_type}")
                    
                    for i, node in enumerate(nodes[:10]):  # Show first 10
                        with st.expander(f"Node {i+1}: {node.get('id', 'Unknown')[:30]}..."):
                            st.write("**Content:**")
                            st.write(node.get('content', 'No content'))
                            st.write("**Metadata:**")
                            st.json(node.get('metadata', {}))
                    
                    if len(nodes) > 10:
                        st.info(f"Showing first 10 of {len(nodes)} nodes. Use API directly for complete results.")
                else:
                    st.error(f"Error: {results['error']}")
    
    with community_tab:
        st.subheader("Graph Communities")
        
        if st.button("🔍 Analyze Communities", key="communities_btn"):
            with st.spinner("Analyzing communities..."):
                api = RapidRFPAPI(st.session_state.api_base_url)
                results = api.get_communities()
                
                if 'error' not in results:
                    communities = results.get('communities', [])
                    st.success(f"Found {len(communities)} communities")
                    
                    for i, community in enumerate(communities):
                        with st.expander(f"Community {i+1} ({community.get('size', 0)} nodes)"):
                            st.write("**Summary:**")
                            st.write(community.get('summary', 'No summary available'))
                            st.write("**Nodes:**")
                            nodes = community.get('nodes', [])
                            for node_id in nodes[:5]:  # Show first 5 nodes
                                st.write(f"• {node_id}")
                            if len(nodes) > 5:
                                st.write(f"... and {len(nodes) - 5} more nodes")
                else:
                    st.error(f"Error: {results['error']}")
    
    with entities_tab:
        st.subheader("Important Entities")
        
        if st.button("⭐ Get Important Entities", key="important_entities_btn"):
            with st.spinner("Analyzing important entities..."):
                api = RapidRFPAPI(st.session_state.api_base_url)
                results = api.get_important_entities()
                
                if 'error' not in results:
                    entities = results.get('entities', [])
                    st.success(f"Found {len(entities)} important entities")
                    
                    for i, entity in enumerate(entities):
                        with st.expander(f"Entity {i+1}: {entity.get('name', 'Unknown')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Importance Score", f"{entity.get('importance_score', 0):.4f}")
                            with col2:
                                st.metric("Mentions", entity.get('mentions', 0))
                            
                            st.write("**Content:**")
                            st.write(entity.get('content', 'No content'))
                else:
                    st.error(f"Error: {results['error']}")

def display_visualization_controls():
    """Display visualization controls."""
    st.header("🎨 Graph Visualization")
    
    api = RapidRFPAPI(st.session_state.api_base_url)
    
    # Visualization options
    viz_tab, community_tab, node_type_tab = st.tabs(["General Viz", "Communities", "Node Types"])
    
    with viz_tab:
        st.subheader("General Graph Visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            max_nodes = st.slider("Max Nodes", 100, 5000, 2000, 100)
            highlight_communities = st.checkbox("Highlight Communities", value=False)
        
        with col2:
            node_types_filter = st.multiselect(
                "Filter Node Types", 
                options=["T", "S", "N", "R", "A", "H", "O"],
                default=[],
                help="Leave empty to include all node types"
            )
        
        if st.button("🌐 Generate General Visualization", type="primary", key="gen_viz"):
            with st.spinner("Creating visualization..."):
                result = api.create_visualization(
                    max_nodes=max_nodes,
                    filter_node_types=node_types_filter if node_types_filter else None,
                    highlight_communities=highlight_communities
                )
                
                if 'error' not in result:
                    st.success("✅ Visualization created successfully!")
                    
                    # Display stats
                    stats = result.get('stats', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nodes", stats.get('total_nodes', 0))
                    with col2:
                        st.metric("Edges", stats.get('total_edges', 0))
                    with col3:
                        st.metric("Communities", stats.get('num_communities', 0))
                    
                    # Show download link
                    filename = result.get('output_filename', 'graph_visualization.html')
                    st.markdown(f"""
                    **📥 Download Visualization:**
                    [Open Visualization]({st.session_state.api_base_url}/api/visualization/serve/{filename})
                    """)
                    
                    # Display parameters used
                    with st.expander("📊 Visualization Parameters"):
                        st.json(result.get('parameters', {}))
                    
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    with community_tab:
        st.subheader("Community-Focused Visualization")
        
        st.info("""
        This visualization highlights different communities in the graph using distinct colors,
        making it easy to identify clusters and relationships between different topic areas.
        """)
        
        if st.button("🌐 Generate Community Visualization", type="primary", key="community_viz"):
            with st.spinner("Creating community visualization..."):
                result = api.create_community_visualization()
                
                if 'error' not in result:
                    st.success("✅ Community visualization created successfully!")
                    
                    # Display stats
                    st.metric("Communities Found", result.get('communities', 0))
                    
                    # Show download link
                    filename = result.get('output_filename', 'community_visualization.html')
                    st.markdown(f"""
                    **📥 Download Community Visualization:**
                    [Open Visualization]({st.session_state.api_base_url}/api/visualization/serve/{filename})
                    """)
                    
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    with node_type_tab:
        st.subheader("Node Type-Specific Visualization")
        
        selected_types = st.multiselect(
            "Select Node Types to Visualize",
            options=["T", "S", "N", "R", "A", "H", "O"],
            default=["N", "R"],
            help="Choose which node types to include in the visualization"
        )
        
        # Node type descriptions
        type_descriptions = {
            "T": "Text chunks (original document segments)",
            "S": "Semantic units (independent concepts)",
            "N": "Named entities (people, places, organizations)",
            "R": "Relationships (entity connections)",
            "A": "Attributes (entity summaries)",
            "H": "High-level summaries (community overviews)",
            "O": "Overview titles (keyword-based titles)"
        }
        
        if selected_types:
            st.markdown("**Selected Types:**")
            for node_type in selected_types:
                st.write(f"• **{node_type}**: {type_descriptions.get(node_type, 'Unknown')}")
        
        if st.button("🌐 Generate Node Type Visualization", type="primary", 
                     key="node_type_viz", disabled=not selected_types):
            with st.spinner("Creating node type visualization..."):
                result = api.create_node_type_visualization(selected_types)
                
                if 'error' not in result:
                    st.success("✅ Node type visualization created successfully!")
                    
                    # Display stats
                    stats = result.get('stats', {})
                    node_counts = stats.get('node_type_counts', {})
                    
                    st.markdown("**Node Counts by Type:**")
                    for node_type in selected_types:
                        count = node_counts.get(node_type, 0)
                        st.metric(f"Type {node_type}", count)
                    
                    # Show download link
                    filename = result.get('output_filename', 'node_type_visualization.html')
                    st.markdown(f"""
                    **📥 Download Node Type Visualization:**
                    [Open Visualization]({st.session_state.api_base_url}/api/visualization/serve/{filename})
                    """)
                    
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    # Visualization stats
    st.subheader("📊 Visualization Statistics")
    if st.button("🔄 Get Visualization Stats", key="viz_stats"):
        with st.spinner("Getting visualization statistics..."):
            result = api.get_visualization_stats()
            
            if 'error' not in result:
                stats = result.get('stats', {})
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Nodes", stats.get('total_nodes', 0))
                with col2:
                    st.metric("Total Edges", stats.get('total_edges', 0))
                with col3:
                    st.metric("Communities", stats.get('num_communities', 0))
                with col4:
                    st.metric("Connected", "✅" if stats.get('is_connected', False) else "❌")
                
                # Graph metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Graph Density", f"{stats.get('density', 0):.4f}")
                with col2:
                    st.metric("Avg Clustering", f"{stats.get('avg_clustering', 0):.4f}")
                
                # Node type breakdown
                node_counts = stats.get('node_type_counts', {})
                if node_counts:
                    st.markdown("**Node Type Distribution:**")
                    for node_type, count in node_counts.items():
                        percentage = (count / stats.get('total_nodes', 1)) * 100
                        st.write(f"• **{node_type}**: {count} nodes ({percentage:.1f}%)")
                
            else:
                st.error(f"❌ Error: {result['error']}")

def main():
    """Main application function."""
    display_header()
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search interface
        display_search_interface()
        
        # Graph analysis
        display_graph_analysis()
    
    with col2:
        # Visualization controls
        display_visualization_controls()
        
        # System status
        st.subheader("📈 System Status")
        if st.session_state.graph_stats and 'error' not in st.session_state.graph_stats:
            stats = st.session_state.graph_stats
            
            # Connection status
            is_connected = stats.get('is_connected', False)
            if is_connected:
                st.success("✅ Graph is connected")
            else:
                st.warning("⚠️ Graph has disconnected components")
            
            # Quick stats
            st.metric("Graph Density", f"{stats.get('density', 0):.4f}")
            st.metric("Avg Clustering", f"{stats.get('avg_clustering', 0):.4f}")

if __name__ == "__main__":
    main()