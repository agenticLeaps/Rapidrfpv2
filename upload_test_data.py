#!/usr/bin/env python3
"""
Upload test data to the knowledge base for testing the generate-response endpoint
"""

import requests
import json
import time

# API configuration
API_BASE_URL = "http://localhost:5001"

def upload_sample_document():
    """Upload a sample document to test the system"""
    
    print("📤 Uploading Sample Document for Testing")
    print("=" * 50)
    
    org_id = "12aff77d-e387-4f4b-93bd-b294756dd96f"  # Same org_id from the user's query
    
    # Sample document content about BeeBee (matching user's query)
    sample_content = """
    # BeeBee Company Executive Summary
    
    ## Company Overview
    BeeBee Corporation is a leading technology company specializing in artificial intelligence and machine learning solutions. Founded in 2018, the company has rapidly grown to become a major player in the AI industry.
    
    ## Key Products and Services
    - AI-powered analytics platform
    - Machine learning consulting services
    - Custom AI model development
    - Data processing and automation tools
    
    ## Financial Performance
    - Annual Revenue: $50 million (2023)
    - Employee Count: 250+ professionals
    - Global Offices: 5 locations across North America and Europe
    
    ## Strategic Initiatives
    BeeBee is focused on expanding its AI capabilities and entering new markets. The company recently secured $15 million in Series B funding to accelerate product development and market expansion.
    
    ## Leadership Team
    - CEO: Sarah Johnson (Former Google AI Director)
    - CTO: Michael Chen (Ex-Microsoft Research)
    - VP of Sales: Lisa Rodriguez (Previously at Salesforce)
    
    ## Competitive Advantages
    - Proprietary machine learning algorithms
    - Strong customer relationships in Fortune 500
    - Experienced technical team with deep AI expertise
    - Scalable cloud-based architecture
    
    ## Future Outlook
    BeeBee plans to double its workforce by 2025 and expand into international markets, particularly in Asia-Pacific region. The company is also investing heavily in research and development of next-generation AI technologies.
    """
    
    # Prepare upload data
    upload_data = {
        "org_id": org_id,
        "user_id": "test_user",
        "content": sample_content,
        "file_name": "beebee_executive_summary.md",
        "content_type": "text/markdown"
    }
    
    try:
        print(f"Uploading document for org_id: {org_id}")
        response = requests.post(
            f"{API_BASE_URL}/api/v1/upload",
            json=upload_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code in [200, 202]:
            result = response.json()
            print("✅ Document uploaded successfully!")
            print(f"   File ID: {result.get('file_id', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   Processing: {result.get('processing', 'N/A')}")
            
            return True
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - make sure NodeRAG service is running on localhost:5001")
        return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def test_after_upload():
    """Test the generate-response endpoint after uploading data"""
    
    print("\n🧪 Testing Generate Response After Upload")
    print("=" * 50)
    
    org_id = "12aff77d-e387-4f4b-93bd-b294756dd96f"
    
    # Wait a moment for processing
    print("⏳ Waiting 5 seconds for document processing...")
    time.sleep(5)
    
    # Test the exact queries that were failing
    test_queries = [
        "hi",
        "tell the executive summary of beebee",
        "what data do you have available",
        "BeeBee company overview",
        "financial performance",
        "leadership team"
    ]
    
    for query in test_queries:
        print(f"\n• Testing query: '{query}'")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/generate-response",
                json={
                    "org_id": org_id,
                    "user_id": "test_user",
                    "query": query,
                    "max_tokens": 800
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Success! Confidence: {result.get('confidence', 0):.2f}")
                print(f"  📊 Sources: {len(result.get('sources', []))}")
                print(f"  🔍 Algorithm: {result.get('algorithm_used', 'N/A')}")
                
                # Show response preview
                response_text = result.get('response', '')
                preview = response_text[:150] + "..." if len(response_text) > 150 else response_text
                print(f"  💬 Response: {preview}")
            else:
                print(f"  ❌ Failed: {response.status_code}")
                print(f"     {response.text}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 NodeRAG Test Data Upload & Validation")
    print("=" * 60)
    
    # Upload sample document
    if upload_sample_document():
        # Test queries after upload
        test_after_upload()
    
    print(f"\n{'='*60}")
    print("✅ Test completed!")
    print("\n💡 Now your knowledge base has data for testing:")
    print("   - Try: 'hi' (should trigger greeting handler)")
    print("   - Try: 'tell the executive summary of beebee'")
    print("   - Try: 'what data do you have available?' (knowledge discovery)")
    print("   - Try: 'BeeBee financial performance'")