#!/usr/bin/env python3
"""
Debug script for the generate-response endpoint issues
"""

import requests
import json

# API configuration
API_BASE_URL = "http://localhost:5001"

def debug_data_availability():
    """Check what data is available in the knowledge base"""
    
    print("🔍 Debugging Data Availability")
    print("=" * 50)
    
    org_id = "12aff77d-e387-4f4b-93bd-b294756dd96f"
    
    # 1. Check if data exists using inspect-data endpoint
    print("1. Checking data with inspect-data endpoint...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/inspect-data",
            json={"org_id": org_id},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Found {result.get('total_groups', 0)} data groups")
            
            if result.get('results'):
                for i, item in enumerate(result['results'][:5]):  # Show first 5
                    print(f"   📁 {i+1}. File: {item.get('file_id', 'N/A')}")
                    print(f"      - Node type: {item.get('node_type', 'N/A')}")
                    print(f"      - Count: {item.get('count', 0)}")
                    print(f"      - Last created: {item.get('last_created', 'N/A')}")
            else:
                print("   ❌ No data found for this org_id")
        else:
            print(f"   ❌ Inspect failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def debug_search_functionality():
    """Test search functionality with different queries"""
    
    print("\n🔍 Testing Search Functionality")
    print("=" * 50)
    
    org_id = "12aff77d-e387-4f4b-93bd-b294756dd96f"
    
    # Test queries from simple to complex
    test_queries = [
        "hi",
        "hello",
        "what data do you have",
        "overview",
        "summary",
        "information",
        "companies",
        "organizations",
        "main topics"
    ]
    
    for query in test_queries:
        print(f"\n• Testing query: '{query}'")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/search",
                json={
                    "org_id": org_id,
                    "query": query,
                    "top_k": 5
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                count = result.get('count', 0)
                print(f"  📊 Found {count} results")
                
                if count > 0:
                    # Show first result
                    first_result = result['results'][0]
                    print(f"  📄 Top result: {first_result.get('content', '')[:100]}...")
                    print(f"  🎯 Similarity: {first_result.get('similarity_score', 0):.3f}")
                    print(f"  📝 Node type: {first_result.get('node_type', 'N/A')}")
            else:
                print(f"  ❌ Search failed: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_generate_response_detailed():
    """Test generate-response with detailed debugging"""
    
    print("\n🔍 Testing Generate Response with Debug Info")
    print("=" * 50)
    
    org_id = "12aff77d-e387-4f4b-93bd-b294756dd96f"
    
    # Test with knowledge discovery query first
    test_data = {
        "org_id": org_id,
        "user_id": "debug_user",
        "query": "what data do you have available",  # Should trigger knowledge discovery
        "max_tokens": 1024,
        "temperature": 0.7
    }
    
    print("Testing with knowledge discovery query...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/generate-response",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response generated successfully!")
            print(f"   Algorithm: {result.get('algorithm_used', 'N/A')}")
            print(f"   Agentic mode: {result.get('agentic_mode', False)}")
            print(f"   Context used: {result.get('context_used', 0)}")
            print(f"   Sources: {len(result.get('sources', []))}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            
            if 'retrieval_metadata' in result:
                metadata = result['retrieval_metadata']
                print(f"   Storage search count: {metadata.get('storage_search_count', 0)}")
                print(f"   Node types: {metadata.get('node_type_distribution', {})}")
            
            # Show response preview
            response_text = result.get('response', '')
            preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
            print(f"   Response: {preview}")
        else:
            print(f"❌ Generate response failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_service_health():
    """Check if the service is running and healthy"""
    
    print("\n🔍 Checking Service Health")
    print("=" * 30)
    
    try:
        # Try a simple endpoint
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Service is running")
        else:
            print(f"⚠️  Service responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Service is not running or not accessible")
        print("   Start with: python api_service.py")
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("🐛 NodeRAG Generate Response Debugging Tool")
    print("=" * 60)
    
    # Run diagnostics
    check_service_health()
    debug_data_availability()
    debug_search_functionality() 
    test_generate_response_detailed()
    
    print(f"\n{'='*60}")
    print("🔧 Debug Summary Complete!")
    print("\n💡 Next Steps:")
    print("1. If no data found: Upload documents using /api/v1/upload")
    print("2. If search fails: Check Neo4j connection and storage")
    print("3. If generate-response fails: Check LLM service configuration")