#!/usr/bin/env python3
"""
Local Queue Testing Script for NodeRAG
Tests the queue system locally before Render deployment
"""

import requests
import time
import json
import subprocess
import signal
import os
import sys
from typing import List, Dict, Any

def check_redis():
    """Check if Redis is running"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis is running")
        return True
    except Exception as e:
        print(f"❌ Redis not running: {e}")
        print("💡 Start Redis with: redis-server")
        return False

def check_services():
    """Check if required services are running"""
    services = {
        "Redis": check_redis(),
    }
    
    # Check API server
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            services["API"] = True
        else:
            print(f"❌ API server returned {response.status_code}")
            services["API"] = False
    except requests.exceptions.RequestException:
        print("❌ API server not running")
        services["API"] = False
    
    # Check queue stats
    try:
        response = requests.get("http://localhost:8000/api/v1/queue/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Queue system is running - Workers: {stats.get('workers_available', 0)}")
            services["Queue"] = True
        else:
            print("❌ Queue system not responding")
            services["Queue"] = False
    except requests.exceptions.RequestException:
        print("❌ Queue system not running")
        services["Queue"] = False
    
    return services

def submit_test_document(org_id: str, file_id: str, use_queue: bool = True) -> Dict[str, Any]:
    """Submit a test document for processing"""
    
    test_chunks = [
        {
            "content": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "metadata": {"page": 1, "section": "introduction"}
        },
        {
            "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
            "metadata": {"page": 1, "section": "deep_learning"}
        },
        {
            "content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
            "metadata": {"page": 2, "section": "nlp"}
        }
    ]
    
    payload = {
        "org_id": org_id,
        "file_id": file_id,
        "user_id": "test-user",
        "chunks": test_chunks,
        "use_queue": use_queue
    }
    
    try:
        print(f"📤 Submitting document {file_id} (queue: {use_queue})")
        response = requests.post(
            "http://localhost:8000/api/v1/process-document",
            json=payload,
            timeout=10
        )
        
        if response.status_code in [200, 202]:
            result = response.json()
            print(f"✅ Document submitted successfully")
            print(f"   Task ID: {result.get('task_id', 'N/A')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Queue Mode: {result.get('queue_mode', False)}")
            return result
        else:
            print(f"❌ Failed to submit document: {response.status_code}")
            print(f"   Response: {response.text}")
            return {"error": response.text}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {"error": str(e)}

def monitor_task_status(file_id: str, max_wait: int = 300) -> Dict[str, Any]:
    """Monitor task processing status"""
    print(f"🔍 Monitoring task status for {file_id}")
    
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"http://localhost:8000/api/v1/status/{file_id}", timeout=5)
            
            if response.status_code == 200:
                status = response.json()
                current_status = status.get('status')
                phase = status.get('phase', 'unknown')
                progress = status.get('progress', 0)
                
                # Only print if status changed
                if current_status != last_status:
                    print(f"📊 Status: {current_status} | Phase: {phase} | Progress: {progress}%")
                    last_status = current_status
                
                # Add memory info if available
                memory_mb = status.get('memory_mb')
                if memory_mb:
                    print(f"   Memory Usage: {memory_mb:.1f} MB")
                
                # Check if completed
                if current_status in ['completed', 'failed', 'skipped']:
                    print(f"🏁 Task finished with status: {current_status}")
                    if current_status == 'completed':
                        results = status.get('results', {})
                        print(f"   Results: {json.dumps(results, indent=2)}")
                    elif current_status == 'failed':
                        error = status.get('error', 'Unknown error')
                        print(f"   Error: {error}")
                    return status
                
            else:
                print(f"❌ Status check failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Status check error: {e}")
        
        time.sleep(5)  # Check every 5 seconds
    
    print(f"⏰ Timeout after {max_wait} seconds")
    return {"error": "timeout"}

def test_queue_system():
    """Main test function"""
    print("🧪 Testing NodeRAG Queue System Locally")
    print("=" * 50)
    
    # Step 1: Check services
    print("\n1. Checking Services...")
    services = check_services()
    
    if not all(services.values()):
        print("\n❌ Some services are not running. Please start them:")
        if not services.get("Redis"):
            print("   Start Redis: redis-server")
        if not services.get("API"):
            print("   Start API: python api_service.py")
        if not services.get("Queue"):
            print("   Start Worker: python start_noderag_service.py (with NODERAG_SERVICE_TYPE=worker)")
        return False
    
    print("\n✅ All services are running!")
    
    # Step 2: Test queue mode
    print("\n2. Testing Queue Mode...")
    org_id = "test-org"
    file_id = f"test-file-{int(time.time())}"
    
    result = submit_test_document(org_id, file_id, use_queue=True)
    if "error" in result:
        return False
    
    # Monitor the task
    status = monitor_task_status(file_id)
    
    if status.get('status') == 'completed':
        print("✅ Queue mode test passed!")
        return True
    else:
        print("❌ Queue mode test failed!")
        return False

def test_concurrent_requests():
    """Test multiple concurrent requests"""
    print("\n3. Testing Concurrent Requests (5 documents)...")
    
    org_id = "test-org"
    base_timestamp = int(time.time())
    tasks = []
    
    # Submit 5 documents concurrently
    for i in range(5):
        file_id = f"concurrent-test-{base_timestamp}-{i}"
        result = submit_test_document(org_id, file_id, use_queue=True)
        if "task_id" in result:
            tasks.append({"file_id": file_id, "task_id": result["task_id"]})
        time.sleep(0.5)  # Small delay between submissions
    
    print(f"📤 Submitted {len(tasks)} documents")
    
    # Monitor all tasks
    completed = 0
    failed = 0
    
    for task in tasks:
        print(f"\n🔍 Monitoring {task['file_id']}...")
        status = monitor_task_status(task['file_id'], max_wait=180)  # 3 minutes per task
        
        if status.get('status') == 'completed':
            completed += 1
        else:
            failed += 1
    
    print(f"\n📊 Concurrent Test Results:")
    print(f"   ✅ Completed: {completed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%)")
    
    return completed == len(tasks)

def main():
    """Main function"""
    print("🚀 NodeRAG Queue System Local Testing")
    print("This will test the queue system on your local machine")
    print("Make sure you have started the required services first!")
    print()
    
    # Set environment for local testing
    os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
    os.environ['PYTHONPATH'] = '.'
    
    try:
        # Run basic queue test
        if test_queue_system():
            print("\n🎉 Basic queue test passed!")
        else:
            print("\n❌ Basic queue test failed!")
            return 1
        
        # Ask user if they want to test concurrent requests
        response = input("\n❓ Test concurrent requests? (y/n): ").lower().strip()
        if response == 'y':
            if test_concurrent_requests():
                print("\n🎉 Concurrent test passed!")
            else:
                print("\n❌ Concurrent test failed!")
        
        print("\n✅ Local queue testing completed!")
        print("\n💡 Next steps:")
        print("   1. Review the logs for any issues")
        print("   2. Deploy to Render using render.yaml")
        print("   3. Test on production environment")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())