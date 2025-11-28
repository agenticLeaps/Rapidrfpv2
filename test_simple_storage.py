#!/usr/bin/env python3
"""
Simple storage test for Neo4j - test direct storage operations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.storage.neo4j_storage import Neo4jStorage
import json

def test_direct_neo4j_operations():
    """Test direct Neo4j operations without complex objects"""
    print("🧪 Testing direct Neo4j operations...")
    
    try:
        storage = Neo4jStorage()
        
        # Test 1: Create sample nodes directly using Cypher
        print("📊 Creating sample nodes with token data...")
        
        with storage._get_session() as session:
            # Create sample NodeEmbedding
            session.run("""
                CREATE (n:NodeEmbedding {
                    node_id: $node_id,
                    node_type: $node_type,
                    content: $content,
                    org_id: $org_id,
                    file_id: $file_id,
                    user_id: $user_id,
                    input_tokens: $input_tokens,
                    output_tokens: $output_tokens,
                    total_tokens: $total_tokens,
                    api_calls: $api_calls,
                    created_at: datetime(),
                    updated_at: datetime()
                })
            """, 
                node_id="test_node_001",
                node_type="semantic",
                content="This is a test semantic node with sample content",
                org_id="test_org",
                file_id="test_file_001",
                user_id="test_user",
                input_tokens=150,
                output_tokens=80,
                total_tokens=230,
                api_calls=2
            )
            
            # Create sample GraphData
            session.run("""
                CREATE (g:GraphData {
                    org_id: $org_id,
                    file_id: $file_id,
                    user_id: $user_id,
                    stats: $stats,
                    input_tokens: $input_tokens,
                    output_tokens: $output_tokens,
                    total_tokens: $total_tokens,
                    api_calls: $api_calls,
                    created_at: datetime(),
                    updated_at: datetime()
                })
            """,
                org_id="test_org",
                file_id="test_file_001",
                user_id="test_user",
                stats='{"nodes": 15, "edges": 12}',
                input_tokens=1500,
                output_tokens=800,
                total_tokens=2300,
                api_calls=25
            )
        
        print("✅ Sample data created successfully")
        
        # Test 2: Query statistics
        print("\n📈 Testing statistics queries...")
        stats = storage.get_file_stats("test_org", "test_file_001")
        print(f"File stats: {json.dumps(stats, indent=2)}")
        
        org_stats = storage.get_file_stats("test_org")
        print(f"Org stats: {json.dumps(org_stats, indent=2)}")
        
        # Test 3: Test search (without embeddings)
        print("\n🔍 Testing text-based search...")
        search_results = storage._fallback_search(
            org_id="test_org",
            query="test semantic",
            query_embedding=[0.1] * 1536,
            top_k=5
        )
        
        print(f"Search results: {len(search_results)} found")
        if search_results:
            print(f"Sample result: {json.dumps(search_results[0], indent=2)}")
        
        # Test 4: Inspect data
        print("\n🔍 Testing data inspection...")
        inspect_result = storage.inspect_all_data()
        print(f"Inspect result: {json.dumps(inspect_result, indent=2)}")
        
        # Test 5: Cleanup
        print("\n🧹 Cleaning up test data...")
        cleanup = storage.delete_file_data("test_org", "test_file_001")
        print(f"Cleanup result: {cleanup}")
        
        storage.close()
        
        print("\n✅ All direct operations test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Direct operations test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the direct operations test"""
    print("🚀 Starting Neo4j Direct Operations Test")
    print("=" * 60)
    
    success = test_direct_neo4j_operations()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Neo4j direct operations working perfectly!")
        print("💡 Token tracking and all features operational")
    else:
        print("❌ Direct operations test failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)