#!/usr/bin/env python3
"""
Test script for the enhanced /api/v1/generate-response endpoint
"""

import requests
import json
import time

# API configuration
API_BASE_URL = "http://localhost:5001"

def test_generate_response():
    """Test the enhanced generate response endpoint"""
    
    print("🧪 Testing Enhanced Generate Response API")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "name": "Knowledge Discovery Query",
            "data": {
                "org_id": "test_org_123",
                "user_id": "test_user_456", 
                "query": "What data do you have? Give me an overview of available information.",
                "max_tokens": 1024,
                "temperature": 0.7
            }
        },
        {
            "name": "Specific Query",
            "data": {
                "org_id": "test_org_123",
                "user_id": "test_user_456",
                "query": "Tell me about the main companies and organizations mentioned in the documents",
                "max_tokens": 800,
                "temperature": 0.5
            }
        },
        {
            "name": "Query with Conversation History",
            "data": {
                "org_id": "test_org_123",
                "user_id": "test_user_456",
                "query": "What are the key processes described?",
                "conversation_history": "User asked about companies. Assistant provided list of organizations.",
                "max_tokens": 600,
                "temperature": 0.6
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/api/v1/generate-response",
                json=test_case["data"],
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ Response generated successfully!")
                print(f"   Processing time: {processing_time:.2f}s")
                print(f"   Response length: {len(result['response'])} chars")
                print(f"   Sources found: {result.get('context_used', 0)}")
                print(f"   Algorithm: {result.get('algorithm_used', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                
                # Token usage information
                if 'token_usage' in result:
                    token_info = result['token_usage']
                    print(f"   Token usage: {token_info.get('input_tokens', 0)} in, {token_info.get('output_tokens', 0)} out, {token_info.get('total_tokens', 0)} total")
                    print(f"   API calls: {token_info.get('api_calls', 0)}")
                
                # Quality metrics
                if 'quality_metrics' in result:
                    quality = result['quality_metrics']
                    print(f"   Quality score: {quality.get('quality_score', 0):.3f}")
                    print(f"   Response length score: {quality.get('response_length_score', 0):.3f}")
                    print(f"   Source diversity: {quality.get('source_diversity_score', 0):.3f}")
                    print(f"   Retrieval coverage: {quality.get('retrieval_coverage', 0):.3f}")
                
                # Response preview
                response_preview = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
                print(f"   Response preview: {response_preview}")
                
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection failed - make sure NodeRAG service is running on localhost:5001")
            print("   Start with: python api_service.py")
        except requests.exceptions.Timeout:
            print("❌ Request timed out (>60s)")
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n{'='*60}")
    print("✅ Generate Response API Test Suite Completed!")

def test_error_conditions():
    """Test error handling"""
    
    print("\n🧪 Testing Error Conditions")
    print("=" * 40)
    
    error_tests = [
        {
            "name": "Missing org_id",
            "data": {"query": "test query"},
            "expected_status": 400
        },
        {
            "name": "Missing query", 
            "data": {"org_id": "test_org"},
            "expected_status": 400
        },
        {
            "name": "Empty query",
            "data": {"org_id": "test_org", "query": ""},
            "expected_status": 400
        },
        {
            "name": "Query too long",
            "data": {"org_id": "test_org", "query": "x" * 6000},
            "expected_status": 400
        },
        {
            "name": "Invalid temperature",
            "data": {"org_id": "test_org", "query": "test", "temperature": 3.0},
            "expected_status": 400
        }
    ]
    
    for test in error_tests:
        print(f"\n• Testing: {test['name']}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/generate-response",
                json=test["data"],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == test["expected_status"]:
                print(f"  ✅ Correct status code: {response.status_code}")
            else:
                print(f"  ❌ Wrong status code: {response.status_code} (expected {test['expected_status']})")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 NodeRAG Enhanced Generate Response Test Suite")
    print("=" * 80)
    
    # Test main functionality
    test_generate_response()
    
    # Test error handling  
    test_error_conditions()
    
    print(f"\n{'='*80}")
    print("📊 Test Summary:")
    print("✅ All tests completed!")
    print("\n💡 Usage Examples:")
    print("""
1. Basic query:
curl -X POST http://localhost:5001/api/v1/generate-response \\
  -H "Content-Type: application/json" \\
  -d '{
    "org_id": "org_123",
    "user_id": "user_456",
    "query": "What information do you have available?"
  }'

2. Query with parameters:
curl -X POST http://localhost:5001/api/v1/generate-response \\
  -H "Content-Type: application/json" \\
  -d '{
    "org_id": "org_123", 
    "user_id": "user_456",
    "query": "Tell me about the companies mentioned",
    "max_tokens": 1024,
    "temperature": 0.7,
    "conversation_history": "Previous context here..."
  }'
    """)