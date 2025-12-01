#!/usr/bin/env python3
"""
Database Performance Test Script
Tests the optimized NeonDB storage performance improvements
"""

import asyncio
import time
import json
import random
import string
from typing import List, Dict
from src.storage.neon_storage import NeonDBStorage

def generate_test_data(num_embeddings: int = 100) -> List[Dict]:
    """Generate test embedding data"""
    test_data = []
    
    for i in range(num_embeddings):
        # Generate random embedding (1536 dimensions like OpenAI)
        embedding = [random.random() for _ in range(1536)]
        
        test_data.append({
            'node_id': f'test_node_{i}',
            'node_type': random.choice(['S', 'N', 'R', 'A', 'H', 'O']),
            'content': f'Test content for node {i} with some random text: ' + ''.join(random.choices(string.ascii_letters + ' ', k=200)),
            'embedding': embedding,
            'metadata': {
                'test_batch': True,
                'node_index': i,
                'created_time': time.time()
            }
        })
    
    return test_data

async def test_performance():
    """Test database performance improvements"""
    print("🚀 Starting database performance tests...")
    
    try:
        # Initialize storage
        storage = NeonDBStorage()
        
        test_org_id = "test_performance_org"
        test_file_id = f"test_file_{int(time.time())}"
        test_user_id = "test_user"
        
        # Test different data sizes
        test_sizes = [50, 100, 500, 1000]
        
        for size in test_sizes:
            print(f"\n📊 Testing with {size} embeddings...")
            
            # Generate test data
            test_data = generate_test_data(size)
            print(f"✅ Generated {len(test_data)} test embeddings")
            
            # Test bulk storage
            start_time = time.time()
            result = storage.bulk_store_embeddings(
                org_id=test_org_id,
                file_id=f"{test_file_id}_{size}",
                user_id=test_user_id,
                embedding_data=test_data
            )
            total_time = time.time() - start_time
            
            if result.get("success"):
                throughput = result.get("throughput_ops_per_sec", 0)
                print(f"✅ Stored {result['embeddings_stored']} embeddings in {total_time:.2f}s")
                print(f"📈 Throughput: {throughput:.1f} operations/second")
            else:
                print(f"❌ Storage failed: {result.get('error', 'Unknown error')}")
                continue
            
            # Test search performance
            test_queries = [
                "test content random",
                "node information",
                "sample data text"
            ]
            
            for query in test_queries:
                search_start = time.time()
                search_results = storage.search_noderag_data(
                    org_id=test_org_id,
                    query=query,
                    top_k=10
                )
                search_time = time.time() - search_start
                
                print(f"🔍 Search '{query[:20]}...': {len(search_results)} results in {search_time:.3f}s")
            
            # Test deletion performance
            delete_start = time.time()
            delete_result = storage.delete_file_data(
                org_id=test_org_id,
                file_id=f"{test_file_id}_{size}"
            )
            delete_time = time.time() - delete_start
            
            if delete_result.get("success"):
                print(f"🗑️ Deleted {delete_result['deleted_count']} embeddings in {delete_time:.3f}s")
            else:
                print(f"❌ Deletion failed: {delete_result.get('error', 'Unknown error')}")
        
        # Close connection pool
        await storage.close_pool()
        print("\n✅ Database performance tests completed!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run performance tests"""
    print("Database Performance Test")
    print("=" * 50)
    
    # Run async test
    asyncio.run(test_performance())

if __name__ == "__main__":
    main()