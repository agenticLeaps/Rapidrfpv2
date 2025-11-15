import streamlit as st
import requests
from typing import Dict

# Configure page
st.set_page_config(
    page_title="RapidRFP RAG System",
    page_icon="🔗",
    layout="wide"
)

# Initialize session state
if "api_base_url" not in st.session_state:
    try:
        st.session_state.api_base_url = st.secrets.get("API_BASE_URL", "http://localhost:5001")
    except:
        st.session_state.api_base_url = "http://localhost:5001"

if "last_upload_status" not in st.session_state:
    st.session_state.last_upload_status = None

class RapidRFPAPI:
    """Simple API client for RapidRFP RAG system."""
    
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
                'details': str(e)
            }
    
    def health_check(self) -> Dict:
        """Check API health."""
        return self._make_request('GET', '/health')
    
    def upload_document(self, file) -> Dict:
        """Upload and index a document."""
        return self._make_request('POST', '/api/upload/document', files={'file': file})
    
    def get_graph_stats(self) -> Dict:
        """Get graph statistics."""
        return self._make_request('GET', '/api/graph/stats')
    
    def create_visualization(self, max_nodes: int = 1000) -> Dict:
        """Create graph visualization with optimal settings."""
        return self._make_request('POST', '/api/visualization/create', {
            'max_nodes': max_nodes,
            'highlight_communities': True,
            'output_filename': 'graph_visualization.html'
        })
    
    def answer_query(self, query: str) -> Dict:
        """Generate comprehensive answer with optimal settings."""
        return self._make_request('POST', '/api/answer', {
            'query': query,
            'use_structured_prompt': True,
            'k_hnsw': 15,
            'k_final': 30,
            'entity_limit': 15,
            'relationship_limit': 40,
            'high_level_limit': 15,
            'include_sources': True,
            'include_reasoning': True
        })
    
    def get_nodes_by_type(self, node_type: str) -> Dict:
        """Get nodes by type."""
        return self._make_request('GET', f'/api/graph/nodes/{node_type}')
    
    def debug_knowledge_base(self) -> Dict:
        """Debug knowledge base status."""
        return self._make_request('GET', '/api/debug/knowledge-base')

def main():
    """Main application function."""
    
    # Header
    st.title("🔗 RapidRFP RAG System")
    st.markdown("**Intelligent Document Analysis and Q&A System**")
    
    # API Status Check
    api = RapidRFPAPI(st.session_state.api_base_url)
    health = api.health_check()
    
    if 'error' not in health:
        st.success(f"✅ System Ready")
    else:
        st.error(f"❌ System Error: {health.get('error', 'Unknown')}")
        st.stop()
    
    # Main layout with three columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Column 1: Document Upload
    with col1:
        st.subheader("📁 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'docx', 'txt', 'md'],
            help="Upload a document to analyze"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing document..."):
                    result = api.upload_document(uploaded_file)
                    
                    if 'error' not in result:
                        st.success("✅ Document processed successfully!")
                        st.session_state.last_upload_status = {
                            'success': True,
                            'filename': uploaded_file.name,
                            'details': result
                        }
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result['error']}")
                        st.session_state.last_upload_status = {
                            'success': False,
                            'error': result['error']
                        }
        
        # Show last upload status
        if st.session_state.last_upload_status:
            if st.session_state.last_upload_status['success']:
                st.info(f"📄 Last processed: {st.session_state.last_upload_status['filename']}")
    
    # Column 2: Graph Visualization
    with col2:
        st.subheader("🎨 Knowledge Graph")
        
        # Get graph stats
        stats = api.get_graph_stats()
        if 'error' not in stats:
            st.metric("Total Nodes", stats.get('total_nodes', 0))
            st.metric("Connections", stats.get('total_edges', 0))
            
            if stats.get('total_nodes', 0) > 0:
                if st.button("🌐 Generate Visualization", type="secondary", use_container_width=True):
                    with st.spinner("Creating visualization..."):
                        viz_result = api.create_visualization()
                        
                        if 'error' not in viz_result and viz_result.get('success', False):
                            st.success("✅ Visualization created!")
                            filename = viz_result.get('output_filename', 'graph_visualization.html')
                            viz_url = f"{st.session_state.api_base_url}/api/visualization/serve/{filename}"
                            st.markdown(f"""
                            **📥 <a href="{viz_url}" target="_blank">Open Visualization in New Tab</a>**
                            """, unsafe_allow_html=True)
                            st.info(f"📋 Or copy this URL: {viz_url}")
                        else:
                            error_msg = viz_result.get('error', 'Unknown error occurred')
                            st.error(f"❌ Error creating visualization: {error_msg}")
                            if 'details' in viz_result:
                                with st.expander("Error Details"):
                                    st.code(viz_result['details'])
            else:
                st.info("Upload a document first to generate visualization")
        else:
            st.error("Cannot load graph statistics")
    
    # Column 3: Quick Stats
    with col3:
        st.subheader("📊 System Stats")
        if 'error' not in stats:
            node_counts = stats.get('node_type_counts', {})
            if node_counts:
                st.write("**Knowledge Types:**")
                type_descriptions = {
                    "N": f"Entities: {node_counts.get('N', 0)}",
                    "R": f"Relations: {node_counts.get('R', 0)}",
                    "T": f"Text Chunks: {node_counts.get('T', 0)}",
                    "H": f"Summaries: {node_counts.get('H', 0)}"
                }
                for desc in type_descriptions.values():
                    if int(desc.split(': ')[1]) > 0:
                        st.write(f"• {desc}")
            
            # Graph quality metrics
            if stats.get('total_nodes', 0) > 0:
                density = stats.get('density', 0)
                if density > 0:
                    st.metric("Graph Density", f"{density:.3f}")
                
                communities = stats.get('num_communities', 0)
                if communities > 0:
                    st.metric("Topic Clusters", communities)
    
    # Main Q&A Section
    st.markdown("---")
    st.subheader("🤖 Ask Questions About Your Documents")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="Ask a detailed question about your uploaded documents...",
        height=100
    )
    
    # Generate answer
    if st.button("🔍 Get Answer", type="primary", disabled=not query):
        if stats.get('total_nodes', 0) == 0:
            st.warning("⚠️ Please upload and process a document first.")
        else:
            with st.spinner("🔍 Analyzing documents and generating answer..."):
                result = api.answer_query(query)
                
                if result.get('success'):
                    # Display answer
                    st.markdown("### 🎯 Answer")
                    st.info(result['answer'])
                    
                    # Display sources and reasoning
                    if 'retrieved_nodes' in result and result['retrieved_nodes']:
                        st.markdown("### 📚 Sources & Evidence")
                        
                        # Group nodes by type for better display
                        sources_by_type = {}
                        for node in result['retrieved_nodes']:
                            node_type = node.get('type', 'Unknown')
                            if node_type not in sources_by_type:
                                sources_by_type[node_type] = []
                            sources_by_type[node_type].append(node)
                        
                        # Display sources in tabs
                        if 'T' in sources_by_type:  # Text chunks - most important
                            st.markdown("**📄 Source Texts:**")
                            for i, chunk in enumerate(sources_by_type['T'][:5]):  # Show top 5
                                with st.expander(f"Source {i+1} - {chunk['id'][:30]}..."):
                                    st.write(chunk['content'])
                        
                        if 'N' in sources_by_type:  # Named entities
                            st.markdown("**🏷️ Key Entities:**")
                            entity_names = [node['content'][:50] for node in sources_by_type['N'][:10]]
                            st.write(", ".join(entity_names))
                        
                        if 'R' in sources_by_type:  # Relationships
                            st.markdown("**🔗 Key Relationships:**")
                            for rel in sources_by_type['R'][:3]:
                                st.write(f"• {rel['content'][:100]}...")
                    
                    # Display retrieval metadata
                    if 'retrieval_metadata' in result:
                        metadata = result['retrieval_metadata']
                        st.markdown("### 📊 Analysis Details")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sources Found", metadata.get('total_nodes_retrieved', 0))
                        with col2:
                            st.metric("Exact Matches", metadata.get('exact_matches', 0))
                        with col3:
                            st.metric("Similar Content", metadata.get('hnsw_results', 0))
                
                else:
                    st.error(f"❌ Error generating answer: {result.get('error', 'Unknown error')}")
                    if 'traceback' in result:
                        with st.expander("Error Details"):
                            st.code(result['traceback'])
    
    # Knowledge Explorer Section
    if stats.get('total_nodes', 0) > 0:
        st.markdown("---")
        st.subheader("🔍 Explore Knowledge")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            node_type = st.selectbox(
                "Knowledge Type:",
                options=["N", "R", "T", "H"],
                format_func=lambda x: {
                    "N": "🏷️ Named Entities", 
                    "R": "🔗 Relationships", 
                    "T": "📄 Text Chunks",
                    "H": "📊 High-level Summaries"
                }.get(x, x)
            )
            
            if st.button("🔍 Explore", type="secondary"):
                with st.spinner("Loading knowledge nodes..."):
                    nodes_result = api.get_nodes_by_type(node_type)
                    
                    if 'error' not in nodes_result:
                        st.session_state['explorer_nodes'] = nodes_result
                    else:
                        st.error(f"Error: {nodes_result['error']}")
        
        with col2:
            if 'explorer_nodes' in st.session_state:
                nodes_data = st.session_state['explorer_nodes']
                st.write(f"**Found {nodes_data.get('count', 0)} {node_type} nodes:**")
                
                # Show first 10 nodes
                nodes = nodes_data.get('nodes', [])[:10]
                for i, node in enumerate(nodes):
                    with st.expander(f"{i+1}. {node['id'][:40]}..."):
                        st.write(f"**Type:** {node['type']}")
                        st.write(f"**Content:** {node['content'][:300]}{'...' if len(node['content']) > 300 else ''}")
                        if node.get('metadata'):
                            st.write(f"**Metadata:** {node['metadata']}")
                
                if len(nodes_data.get('nodes', [])) > 10:
                    st.info(f"Showing first 10 of {nodes_data.get('count', 0)} nodes")
    
    # Debug Section
    st.markdown("---")
    st.subheader("🐛 Debug Information")
    if st.button("🔍 Check Knowledge Base Status", type="secondary"):
        with st.spinner("Checking knowledge base..."):
            debug_result = api.debug_knowledge_base()
            
            if debug_result.get('success'):
                data = debug_result
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Graph Statistics:**")
                    graph_stats = data.get('graph_stats', {})
                    st.write(f"• Total Nodes: {graph_stats.get('total_nodes', 0)}")
                    st.write(f"• Total Edges: {graph_stats.get('total_edges', 0)}")
                    
                    node_counts = graph_stats.get('node_type_counts', {})
                    if node_counts:
                        st.markdown("**Node Types:**")
                        for node_type, count in node_counts.items():
                            st.write(f"• {node_type}: {count}")
                
                with col2:
                    st.markdown("**HNSW Status:**")
                    st.write(data.get('hnsw_status', 'Unknown'))
                    
                    # Show sample nodes
                    samples = data.get('sample_nodes', {})
                    if samples:
                        st.markdown("**Sample Nodes:**")
                        for node_type, nodes in samples.items():
                            if nodes:
                                st.write(f"**{node_type} nodes:** {len(nodes)} samples")
                                for i, node in enumerate(nodes):
                                    st.write(f"  {i+1}. {node['content_preview']}")
                
            else:
                st.error(f"Debug error: {debug_result.get('error', 'Unknown')}")

if __name__ == "__main__":
    main()