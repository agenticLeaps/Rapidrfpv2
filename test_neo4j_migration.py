#!/usr/bin/env python3
"""
Test script for Neo4j migration - verify connectivity and basic operations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.storage.neo4j_storage import Neo4jStorage
import json
import time

def test_neo4j_connection():
    """Test basic Neo4j connectivity"""
    print("🔗 Testing Neo4j connection...")
    
    try:
        storage = Neo4jStorage()
        print("✅ Neo4j connection established successfully")
        
        # Test basic operations
        print("📊 Testing basic database operations...")
        
        # Create constraints and indexes
        storage._ensure_constraints_and_indexes()
        print("✅ Constraints and indexes created/verified")
        
        # Test stats functionality
        stats = storage.get_file_stats("test_org")
        print(f"✅ Stats query successful: {stats}")
        
        # Test inspect functionality
        inspect_result = storage.inspect_all_data()
        print(f"✅ Inspect query successful: Found {inspect_result.get('total_groups', 0)} data groups")
        
        # Clean up test data if any exists
        cleanup = storage.delete_file_data("test_org", "test_file")
        print(f"🧹 Cleanup completed: {cleanup}")
        
        storage.close()
        print("✅ All Neo4j tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Neo4j test failed: {e}")
        return False

def test_performance_comparison():
    """Test performance characteristics"""
    print("\n⚡ Testing performance characteristics...")
    
    try:
        storage = Neo4jStorage()
        
        # Test query performance
        start_time = time.time()
        
        # Simulate multiple rapid queries
        for i in range(5):
            stats = storage.get_file_stats("test_org")
            inspect = storage.inspect_all_data()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        print(f"✅ 10 queries completed in {query_time:.3f}s")
        print(f"📈 Average query time: {query_time/10:.3f}s")
        
        storage.close()
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Neo4j Migration Tests")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    connection_test = test_neo4j_connection()
    
    # Test 2: Performance
    performance_test = test_performance_comparison()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Connection Test: {'✅ PASSED' if connection_test else '❌ FAILED'}")
    print(f"   Performance Test: {'✅ PASSED' if performance_test else '❌ FAILED'}")
    
    if connection_test and performance_test:
        print("\n🎉 All tests passed! Neo4j migration is ready.")
        print("💡 Next steps:")
        print("   1. Update your application to use Neo4jStorage")
        print("   2. Run your existing workflows to verify compatibility")
        print("   3. Monitor performance in production")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)