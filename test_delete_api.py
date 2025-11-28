#!/usr/bin/env python3
"""
Test script for the delete embeddings API functionality
"""

import requests
import json
import time

# API configuration
API_BASE_URL = "http://localhost:5001"

def test_delete_embeddings():
    """Test the delete embeddings endpoint"""
    
    print("🧪 Testing Delete Embeddings API")
    print("=" * 50)
    
    # Test data
    test_data = {
        "org_id": "test_org_123",
        "file_ids": ["file_1", "file_2", "file_3"],
        "callback_url": "http://localhost:5000/api/webhook/noderag-delete"
    }
    
    try:
        # Step 1: Make delete request
        print("📤 Sending delete request...")
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/delete-embeddings",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 202:
            result = response.json()
            print("✅ Delete operation started successfully!")
            print(f"   Delete ID: {result['delete_id']}")
            print(f"   Status: {result['status']}")
            print(f"   File count: {result['file_count']}")
            
            # Step 2: Check status (optional)
            delete_id = result['delete_id']
            print(f"\n🔍 Checking status for delete_id: {delete_id}")
            
            status_response = requests.get(f"{API_BASE_URL}/api/v1/delete-status/{delete_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"   Status: {status_data['status']}")
                print(f"   Message: {status_data['message']}")
            
            print("\n📋 Expected webhook callback:")
            print("   The service should send a POST request to your callback_url with:")
            print("   - status: 'completed' or 'failed'")
            print("   - org_id: Organization ID")
            print("   - file_ids: List of successfully deleted files")
            print("   - deleted_embeddings: Number of embeddings deleted")
            print("   - deleted_graphs: Number of graphs deleted") 
            print("   - success: Boolean indicating overall success")
            
        else:
            print(f"❌ Delete request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - make sure NodeRAG service is running on localhost:5001")
        print("   Start with: python start_noderag_service.py")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

def test_delete_single_file():
    """Test deleting a single file"""
    
    print("\n🧪 Testing Single File Delete")
    print("=" * 50)
    
    test_data = {
        "org_id": "test_org_123",
        "file_id": "single_file_test"
    }
    
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/delete-file",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single file delete successful!")
            print(f"   Message: {result['message']}")
            print(f"   File ID: {result['file_id']}")
            print(f"   Deleted count: {result['deleted_count']}")
        else:
            print(f"❌ Single file delete failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - make sure NodeRAG service is running")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    print("🚀 NodeRAG Delete API Test Suite")
    print("=" * 60)
    
    # Test bulk delete with webhook
    test_delete_embeddings()
    
    # Test single file delete
    test_delete_single_file()
    
    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
    print("\n💡 Usage Examples:")
    print("\n1. Bulk delete with webhook:")
    print("""
    curl -X DELETE http://localhost:5001/api/v1/delete-embeddings \\
      -H "Content-Type: application/json" \\
      -d '{
        "org_id": "org_123",
        "file_ids": ["file_1", "file_2", "file_3"],
        "callback_url": "http://localhost:5000/api/webhook/noderag-delete"
      }'
    """)
    
    print("\n2. Single file delete:")
    print("""
    curl -X DELETE http://localhost:5001/api/v1/delete-file \\
      -H "Content-Type: application/json" \\
      -d '{
        "org_id": "org_123", 
        "file_id": "file_1"
      }'
    """)
    
    print("\n3. Webhook callback format:")
    print("""
    POST http://your-server/webhook/noderag-delete
    {
      "status": "completed",
      "org_id": "org_123",
      "file_ids": ["file_1", "file_2"],
      "deleted_embeddings": 1250,
      "deleted_graphs": 3,
      "success": true,
      "delete_id": "uuid-here",
      "timestamp": 1234567890
    }
    """)