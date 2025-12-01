#!/usr/bin/env python3
"""
Test script for the bulk delete embeddings API
"""

import requests
import json
import time
import sys

def test_bulk_delete_api(base_url="http://localhost:8002"):
    """Test the bulk delete embeddings API endpoint"""
    
    # Test data
    test_payload = {
        "org_id": "test-org-123",
        "file_ids": ["file-1", "file-2", "file-3"],
        "callback_url": "http://localhost:8000/api/webhook/noderag-delete"
    }
    
    print("🧪 Testing bulk delete embeddings API...")
    print(f"📍 Base URL: {base_url}")
    print(f"📋 Test payload: {json.dumps(test_payload, indent=2)}")
    
    try:
        # Test 1: Health check
        print("\n1️⃣ Testing health check...")
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test 2: Bulk delete API
        print("\n2️⃣ Testing bulk delete API...")
        response = requests.delete(
            f"{base_url}/api/v1/delete-embeddings",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 202:
            print("✅ Bulk delete request accepted")
            result = response.json()
            print(f"   Response: {json.dumps(result, indent=2)}")
            
            deletion_id = result.get("deletion_id")
            if deletion_id:
                # Test 3: Check deletion status
                print(f"\n3️⃣ Testing deletion status for ID: {deletion_id}")
                time.sleep(1)  # Give it a moment to start
                
                status_response = requests.get(
                    f"{base_url}/api/v1/delete-status/{deletion_id}",
                    timeout=5
                )
                
                if status_response.status_code == 200:
                    print("✅ Status check passed")
                    print(f"   Status: {json.dumps(status_response.json(), indent=2)}")
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
                    print(f"   Error: {status_response.text}")
            
            return True
            
        else:
            print(f"❌ Bulk delete failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - make sure the API service is running")
        print(f"   Try: python api_service.py")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_error_cases(base_url="http://localhost:8002"):
    """Test error handling"""
    print("\n🚫 Testing error cases...")
    
    # Test missing fields
    print("Testing missing org_id...")
    response = requests.delete(
        f"{base_url}/api/v1/delete-embeddings",
        json={"file_ids": ["test"]},
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    print("✅ Missing org_id handled correctly")
    
    # Test empty file_ids
    print("Testing empty file_ids...")
    response = requests.delete(
        f"{base_url}/api/v1/delete-embeddings",
        json={"org_id": "test", "file_ids": []},
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    print("✅ Empty file_ids handled correctly")
    
    # Test invalid deletion_id
    print("Testing invalid deletion status lookup...")
    response = requests.get(f"{base_url}/api/v1/delete-status/invalid-id")
    assert response.status_code == 404
    print("✅ Invalid deletion_id handled correctly")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8002"
    
    print("🧪 Running API Tests")
    print("=" * 50)
    
    success = test_bulk_delete_api(base_url)
    
    if success:
        test_error_cases(base_url)
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)