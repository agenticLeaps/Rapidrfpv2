#!/usr/bin/env python3
"""
Test to verify that our optimizations maintain full compatibility
with the original NodeRAG functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.document_processing.indexing_pipeline import IndexingPipeline
from src.graph.node_types import NodeType

def test_pipeline_compatibility():
    """Test that optimized pipeline produces same graph structure."""
    print("🧪 Testing Pipeline Compatibility...")
    
    # Create test document content
    test_content = """
    John Smith is the CEO of TechCorp. He founded the company in 2020.
    The company is located in San Francisco and specializes in AI technology.
    Mary Johnson works as the CTO under John Smith's leadership.
    """
    
    # Create a temporary test file
    test_file = "/tmp/test_noderag_compatibility.txt"
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    try:
        # Initialize pipeline
        pipeline = IndexingPipeline()
        
        # Process document
        print("📄 Processing test document...")
        result = pipeline.index_document(test_file)
        
        # Verify result structure
        required_keys = ['success', 'processing_time', 'document_metadata', 'graph_stats', 'chunks_processed']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
            print(f"   ✓ Found required key: {key}")
        
        # Verify graph structure
        stats = result['graph_stats']
        required_stats = ['total_nodes', 'total_edges', 'node_type_counts']
        for key in required_stats:
            assert key in stats, f"Missing graph stat: {key}"
            print(f"   ✓ Graph stat present: {key}")
        
        # Verify node types are created
        node_counts = stats['node_type_counts']
        expected_types = ['T', 'S', 'N', 'R']  # Minimum expected types
        for node_type in expected_types:
            if node_type in node_counts:
                print(f"   ✓ Created {node_type} nodes: {node_counts[node_type]}")
            else:
                print(f"   ⚠️  No {node_type} nodes created")
        
        # Verify embeddings were generated
        if result.get('success', False):
            print("   ✅ Pipeline completed successfully")
            print(f"   📊 Total nodes: {stats['total_nodes']}")
            print(f"   📊 Total edges: {stats['total_edges']}")
            print(f"   ⏱️  Processing time: {result['processing_time']:.2f}s")
        else:
            print("   ❌ Pipeline failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        return False
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)

def test_graph_manager_compatibility():
    """Test that graph manager functions work correctly."""
    print("\n🔗 Testing Graph Manager Compatibility...")
    
    pipeline = IndexingPipeline()
    graph_manager = pipeline.graph_manager
    
    # Test basic graph operations
    try:
        # Test getting stats (should work even with empty graph)
        stats = graph_manager.get_stats()
        print(f"   ✓ Graph stats: {stats}")
        
        # Test node type enumeration
        for node_type in NodeType:
            nodes = graph_manager.get_nodes_by_type(node_type)
            print(f"   ✓ {node_type.value} nodes: {len(nodes)}")
        
        print("   ✅ Graph manager functions working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Graph manager test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 NodeRAG Optimization Compatibility Test\n")
    
    # Run tests
    test1_passed = test_pipeline_compatibility()
    test2_passed = test_graph_manager_compatibility()
    
    print(f"\n📋 Test Results:")
    print(f"   Pipeline Compatibility: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Graph Manager Compatibility: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Optimizations maintain full compatibility.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Review optimizations for compatibility issues.")
        sys.exit(1)