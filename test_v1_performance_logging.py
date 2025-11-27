#!/usr/bin/env python3
"""
Test script for v1 API performance logging.
Demonstrates how to use the /api/v1/process-document endpoint with performance tracking.
"""

import requests
import time
import json
import uuid
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:5000"  # Adjust as needed
ORG_ID = "test_org_123"
USER_ID = "test_user_456"

def generate_test_chunks():
    """Generate sample document chunks for testing."""
    chunks = [
        {
            "content": """
            This is the first chunk about artificial intelligence and machine learning.
            AI technologies are revolutionizing various industries including healthcare,
            finance, and transportation. Key companies in this space include OpenAI,
            Google DeepMind, and Anthropic.
            """,
            "metadata": {"chunk_type": "introduction", "page": 1}
        },
        {
            "content": """
            The second chunk discusses deep learning architectures and neural networks.
            Transformers have become the dominant architecture for natural language processing.
            Important concepts include attention mechanisms, self-attention, and multi-head attention.
            Major applications include language models like GPT, BERT, and Claude.
            """,
            "metadata": {"chunk_type": "technical", "page": 2}
        },
        {
            "content": """
            This third chunk covers the business applications of AI technology.
            Companies are implementing AI for customer service automation, predictive analytics,
            and decision support systems. The market for AI solutions is expected to grow
            significantly over the next decade with investments in autonomous vehicles,
            smart cities, and personalized healthcare.
            """,
            "metadata": {"chunk_type": "business", "page": 3}
        },
        {
            "content": """
            The final chunk addresses ethical considerations and future challenges.
            AI alignment, bias mitigation, and responsible deployment are critical issues.
            Regulatory frameworks are being developed to ensure safe and beneficial AI.
            Collaboration between researchers, policymakers, and industry leaders is essential
            for addressing these challenges effectively.
            """,
            "metadata": {"chunk_type": "conclusion", "page": 4}
        }
    ]
    return chunks

def test_v1_process_document():
    """Test the /api/v1/process-document endpoint with performance logging."""
    
    print("🧪 Testing v1 API Performance Logging")
    print("=" * 60)
    
    # Generate test data
    file_id = f"test_file_{uuid.uuid4().hex[:8]}"
    chunks = generate_test_chunks()
    
    print(f"📁 File ID: {file_id}")
    print(f"📝 Chunks: {len(chunks)}")
    print(f"🏢 Org ID: {ORG_ID}")
    print(f"👤 User ID: {USER_ID}")
    print()
    
    # Step 1: Submit document for processing
    print("🚀 Step 1: Submitting document for processing...")
    
    payload = {
        "org_id": ORG_ID,
        "file_id": file_id,
        "user_id": USER_ID,
        "chunks": chunks,
        "callback_url": None  # No webhook for testing
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/process-document",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 202:
            print(f"❌ Failed to submit document: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        result = response.json()
        print(f"✅ Document submitted successfully")
        print(f"   Status: {result['status']}")
        print(f"   Estimated time: {result['estimated_time']}")
        print()
        
    except Exception as e:
        print(f"❌ Error submitting document: {e}")
        return None
    
    # Step 2: Monitor processing status
    print("📊 Step 2: Monitoring processing status...")
    
    session_id = None
    max_wait_time = 300  # 5 minutes max
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            status_response = requests.get(
                f"{API_BASE_URL}/api/v1/status/{file_id}",
                timeout=10
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                
                print(f"   📈 Status: {status_data['status']} | "
                      f"Phase: {status_data['phase']} | "
                      f"Progress: {status_data['progress']}%")
                
                # Capture session ID when available
                if 'session_id' in status_data and not session_id:
                    session_id = status_data['session_id']
                    print(f"   🔖 Performance Session ID: {session_id}")
                
                # Show real-time performance data if available
                if 'real_time_duration' in status_data:
                    print(f"   ⏱️  Real-time Duration: {status_data['real_time_duration_formatted']}")
                    if status_data.get('current_step'):
                        print(f"   🔄 Current Step: {status_data['current_step']}")
                
                # Check if processing is complete
                if status_data['status'] == 'completed':
                    print(f"\n✅ Processing completed successfully!")
                    
                    # Show performance summary
                    if 'total_duration_formatted' in status_data:
                        print(f"   ⏱️  Total Duration: {status_data['total_duration_formatted']}")
                    
                    if 'processing_rate' in status_data:
                        print(f"   📊 Processing Rate: {status_data['processing_rate']:.1f} bytes/sec")
                    
                    if 'performance_summary' in status_data:
                        summary = status_data['performance_summary']
                        print(f"   📈 Steps: {summary.get('successful_steps', 0)}/{summary.get('total_steps', 0)} successful")
                    
                    print()
                    break
                
                elif status_data['status'] == 'failed':
                    print(f"\n❌ Processing failed: {status_data.get('error', 'Unknown error')}")
                    if 'total_duration' in status_data:
                        print(f"   ⏱️  Failed after: {status_data['total_duration']:.1f}s")
                    return None
                
            time.sleep(2)  # Check every 2 seconds
            
        except Exception as e:
            print(f"   ⚠️  Error checking status: {e}")
            time.sleep(5)
    else:
        print(f"\n⏰ Timeout waiting for processing to complete")
        return None
    
    # Step 3: Get detailed performance report
    if session_id:
        print("📊 Step 3: Retrieving detailed performance report...")
        
        try:
            report_response = requests.get(
                f"{API_BASE_URL}/api/v1/performance/report/{session_id}",
                timeout=10
            )
            
            if report_response.status_code == 200:
                report_data = report_response.json()
                
                if report_data['success']:
                    report = report_data['report']
                    
                    print("📈 DETAILED PERFORMANCE REPORT")
                    print("-" * 40)
                    print(f"Session: {report['session_id']}")
                    print(f"File: {report['file_info']['name']}")
                    print(f"Size: {report['file_info']['size_formatted']}")
                    print(f"Duration: {report['timing']['total_duration_formatted']}")
                    print(f"Status: {report['status']}")
                    
                    if 'summary' in report and report['summary']:
                        summary = report['summary']
                        print(f"Processing Rate: {summary['processing_rate']:.1f} bytes/sec")
                        print(f"Steps: {summary['successful_steps']}/{summary['total_steps']} successful")
                    
                    print("\nStep Breakdown:")
                    for step in report['steps']:
                        status_symbol = "✓" if step['status'] == 'completed' else "✗"
                        print(f"  {status_symbol} {step['name']:<30} {step['duration_formatted']:>8}")
                        if step['sub_steps_count'] > 0:
                            print(f"    └─ Sub-steps: {step['sub_steps_count']}")
                    
                    print()
                else:
                    print(f"❌ Failed to get performance report: {report_data['error']}")
            
        except Exception as e:
            print(f"⚠️ Error getting performance report: {e}")
    
    # Step 4: Export performance report to file
    if session_id:
        print("📄 Step 4: Exporting performance report to file...")
        
        try:
            export_response = requests.post(
                f"{API_BASE_URL}/api/v1/performance/export/{session_id}",
                json={},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if export_response.status_code == 200:
                export_data = export_response.json()
                
                if export_data['success']:
                    print(f"✅ Report exported to: {export_data['report_path']}")
                else:
                    print(f"❌ Failed to export report: {export_data['error']}")
            
        except Exception as e:
            print(f"⚠️ Error exporting report: {e}")
    
    # Step 5: Get overall performance statistics
    print("\n📊 Step 5: Getting overall performance statistics...")
    
    try:
        stats_response = requests.get(
            f"{API_BASE_URL}/api/v1/performance/stats",
            timeout=10
        )
        
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            
            if stats_data['success']:
                stats = stats_data['stats']
                
                print("📈 OVERALL PERFORMANCE STATISTICS")
                print("-" * 40)
                print(f"Total Sessions: {stats['total_sessions']}")
                print(f"Completed Sessions: {stats['completed_sessions']}")
                print(f"Success Rate: {stats['success_rate']}%")
                print(f"Average Duration: {stats['average_duration_formatted']}")
                print(f"Recent Sessions (24h): {stats['recent_sessions_24h']}")
                
                if stats['recent_sessions']:
                    print("\nRecent Sessions:")
                    for session in stats['recent_sessions'][:3]:
                        status_icon = "✅" if session['status'] == 'completed' else "❌"
                        duration = f"{session['duration']:.1f}s" if session['duration'] else "N/A"
                        print(f"  {status_icon} {session['file_name']} - {duration}")
            
        print()
        
    except Exception as e:
        print(f"⚠️ Error getting performance statistics: {e}")
    
    return session_id

def test_current_session_monitoring():
    """Test real-time session monitoring during processing."""
    print("\n🔍 Testing Real-time Session Monitoring")
    print("=" * 60)
    
    try:
        current_response = requests.get(
            f"{API_BASE_URL}/api/v1/performance/current",
            timeout=10
        )
        
        if current_response.status_code == 200:
            current_data = current_response.json()
            
            if current_data['success']:
                print("📊 Current Active Session:")
                print(f"   Session ID: {current_data['session_id']}")
                print(f"   File: {current_data['file_name']}")
                print(f"   Duration: {current_data['current_duration_formatted']}")
                print(f"   Current Step: {current_data.get('current_step', 'N/A')}")
                print(f"   Progress: {current_data['steps_completed']}/{current_data['total_steps']} steps")
            else:
                print("ℹ️  No active processing session")
        
    except Exception as e:
        print(f"⚠️ Error getting current session info: {e}")

def main():
    """Main test function."""
    print("V1 API Performance Logging Test")
    print("This script tests the NodeRAG v1 API with comprehensive performance logging")
    print()
    
    # Test health endpoint first
    try:
        health_response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ API service not available at {API_BASE_URL}")
            return
        
        health_data = health_response.json()
        print(f"✅ API service is healthy: {health_data['service']} v{health_data['version']}")
        print()
        
    except Exception as e:
        print(f"❌ Cannot connect to API service at {API_BASE_URL}: {e}")
        print("Make sure the NodeRAG API service is running:")
        print("  python api_service.py")
        return
    
    # Run the main test
    session_id = test_v1_process_document()
    
    if session_id:
        print("🎉 Test completed successfully!")
        print(f"📁 Performance logs available in: data/performance_logs/")
        print(f"🔗 Session ID: {session_id}")
        print("\nYou can now:")
        print(f"1. View the performance report at: GET {API_BASE_URL}/api/v1/performance/report/{session_id}")
        print(f"2. Export the report: POST {API_BASE_URL}/api/v1/performance/export/{session_id}")
        print(f"3. Check overall stats: GET {API_BASE_URL}/api/v1/performance/stats")
        print(f"4. Monitor current session: GET {API_BASE_URL}/api/v1/performance/current")
    else:
        print("❌ Test failed. Check the error messages above.")

if __name__ == "__main__":
    main()