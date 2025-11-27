#!/usr/bin/env python3
"""
Test script for the new /api/v1/process-chunk endpoint
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
CHUNK_ENDPOINT = f"{API_BASE_URL}/api/v1/process-chunk"

def test_chunk_processing():
    """Test the /api/v1/process-chunk endpoint"""
    
    # Test data
    test_chunk = """
    This is a sample chunk of text that will be processed through the NodeRAG pipeline.
    It contains information about artificial intelligence and machine learning concepts.
    The text discusses various algorithms and techniques used in natural language processing.
    
    Key topics include:
    - Text preprocessing and tokenization
    - Semantic analysis and understanding
    - Knowledge graph construction
    - Vector embeddings and similarity search
    
    The goal is to demonstrate how the NodeRAG system can decompose this text into
    meaningful graph nodes and relationships while tracking performance metrics.
    """
    
    # Prepare request data
    request_data = {
        "chunk_text": test_chunk.strip(),
        "chatgpt_model": "gpt-4",
        "chunk_id": f"test_chunk_{int(time.time())}"
    }
    
    print(f"🚀 Testing chunk processing endpoint...")
    print(f"📝 Chunk text length: {len(test_chunk)} characters")
    print(f"🔗 Endpoint: {CHUNK_ENDPOINT}")
    print(f"🆔 Chunk ID: {request_data['chunk_id']}")
    print()
    
    try:
        # Send request
        print("📤 Sending request...")
        start_time = time.time()
        
        response = requests.post(
            CHUNK_ENDPOINT,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        request_duration = end_time - start_time
        
        print(f"⏱️  Request duration: {request_duration:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        print()
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            
            print("✅ SUCCESS! Processing completed")
            print(f"🆔 Session ID: {result.get('session_id', 'N/A')}")
            print(f"📂 Output file: {result.get('output_filename', 'N/A')}")
            print(f"📍 Workspace location: {result.get('workspace_location', 'N/A')}")
            print()
            
            # Print processing summary
            summary = result.get('processing_summary', {})
            print("📈 Processing Summary:")
            print(f"  • Duration: {summary.get('total_duration_formatted', 'N/A')}")
            print(f"  • Graph nodes: {summary.get('graph_nodes', 0)}")
            print(f"  • Graph edges: {summary.get('graph_edges', 0)}")
            print(f"  • Errors: {summary.get('errors_count', 0)}")
            print()
            
            # Show output file location
            output_file = result.get('output_file')
            if output_file:
                print(f"📄 Output saved to: {output_file}")
                print("💡 You can check this file to see the detailed processing results!")
                
        else:
            print("❌ ERROR! Processing failed")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ ERROR! Request timed out (5 minutes)")
    except requests.exceptions.ConnectionError:
        print("🔌 ERROR! Could not connect to the API server")
        print("Make sure the NodeRAG API service is running on localhost:8000")
    except Exception as e:
        print(f"❌ ERROR! Unexpected error: {e}")

def check_api_health():
    """Check if the API service is running"""
    try:
        health_url = f"{API_BASE_URL}/api/v1/health"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API service is running!")
            print(f"   Service: {health_data.get('service', 'N/A')}")
            print(f"   Version: {health_data.get('version', 'N/A')}")
            return True
        else:
            print(f"⚠️ API health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot reach API service: {e}")
        print("Make sure the NodeRAG API service is running on localhost:8000")
        return False

if __name__ == "__main__":
    print("🧪 NodeRAG Chunk Processing Test")
    print("=" * 50)
    print()
    
    # Check API health first
    if check_api_health():
        print()
        test_chunk_processing()
    else:
        print("\n💡 To start the API service, run:")
        print("   python api_service.py")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")