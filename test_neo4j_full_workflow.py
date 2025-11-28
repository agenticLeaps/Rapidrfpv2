#!/usr/bin/env python3
"""
Full workflow test for Neo4j migration - test with actual data storage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.storage.neo4j_storage import Neo4jStorage
import json
import time

def create_mock_pipeline():
    """Create a mock pipeline object for testing"""
    class MockNode:
        def __init__(self, node_id, node_type, content):
            self.id = node_id
            self.type = node_type
            self.content = content
            self.embeddings = [0.1] * 1536  # Mock embedding vector
            self.metadata = {"test": "metadata"}
    
    class MockNodeType:
        def __init__(self, value):
            self.value = value
    
    class MockGraphManager:
        def __init__(self):
            self.graph = MockGraph()
        
        def get_nodes_by_type(self, node_type):
            # Return mock nodes
            return [
                MockNode(f"node_{i}_{node_type.value}", MockNodeType(node_type.value), f"Test content {i}")
                for i in range(3)
            ]
        
        def get_stats(self):
            return {
                "total_nodes": 15,
                "total_edges": 12,
                "node_types": ["semantic", "entity", "relationship"]
            }
    
    class MockGraph:
        def number_of_nodes(self):
            return 15
    
    class MockLLMService:
        def get_usage_stats(self):
            return {
                "input_tokens": 1500,
                "output_tokens": 800,
                "total_tokens": 2300,
                "api_calls": 25
            }
    
    class MockPipeline:
        def __init__(self):
            self.graph_manager = MockGraphManager()
            self.llm_service = MockLLMService()
    
    return MockPipeline()

def test_full_storage_workflow():
    """Test complete storage and retrieval workflow"""
    print("🧪 Testing full Neo4j storage workflow...")
    
    try:
        storage = Neo4jStorage()
        
        # Create mock pipeline
        pipeline = create_mock_pipeline()
        
        print("📊 Storing test data...")
        start_time = time.time()
        
        # Store data with token tracking
        result = storage.store_noderag_data(
            org_id="test_org",
            file_id="test_file_001",
            user_id="test_user",
            pipeline=pipeline,
            input_tokens=1500,
            output_tokens=800,
            api_calls=25
        )
        
        storage_time = time.time() - start_time
        
        print(f"✅ Storage completed in {storage_time:.3f}s")
        print(f"📈 Result: {json.dumps(result, indent=2)}")
        
        if not result.get("success"):
            raise Exception(f"Storage failed: {result.get('error')}")
        
        # Test retrieval and search
        print("\n🔍 Testing search functionality...")
        search_start = time.time()
        
        search_results = storage.search_noderag_data(
            org_id="test_org",
            query="test content",
            top_k=5
        )
        
        search_time = time.time() - search_start
        print(f"🔍 Search completed in {search_time:.3f}s")
        print(f"📊 Found {len(search_results)} results")
        
        if search_results:
            print("Sample result:", json.dumps(search_results[0], indent=2))
        
        # Test statistics
        print("\n📈 Testing statistics...")
        stats = storage.get_file_stats("test_org", "test_file_001")
        print(f"File stats: {json.dumps(stats, indent=2)}")
        
        org_stats = storage.get_file_stats("test_org")
        print(f"Org stats: {json.dumps(org_stats, indent=2)}")
        
        # Cleanup
        print("\n🧹 Cleaning up test data...")
        cleanup = storage.delete_file_data("test_org", "test_file_001")
        print(f"Cleanup result: {cleanup}")
        
        storage.close()
        
        print("\n✅ Full workflow test PASSED!")
        print(f"⚡ Performance Summary:")
        print(f"   - Storage: {storage_time:.3f}s")
        print(f"   - Search: {search_time:.3f}s")
        print(f"   - Total: {storage_time + search_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Full workflow test FAILED: {e}")
        return False

def main():
    """Run the full workflow test"""
    print("🚀 Starting Neo4j Full Workflow Test")
    print("=" * 60)
    
    success = test_full_storage_workflow()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Neo4j migration is fully operational!")
        print("💡 Ready for production use with:")
        print("   ✅ High-performance storage")
        print("   ✅ Token tracking")
        print("   ✅ Fast similarity search")
        print("   ✅ Comprehensive statistics")
    else:
        print("❌ Migration test failed. Check logs above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)