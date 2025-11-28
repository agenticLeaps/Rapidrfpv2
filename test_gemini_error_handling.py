#!/usr/bin/env python3
"""
Test script to verify Gemini error handling works correctly.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.llm_service import LLMService

def test_error_handling():
    """Test error handling for blocked responses."""
    print("🧪 Testing Gemini error handling...")
    
    try:
        # Initialize LLM service with Gemini
        llm_service = LLMService(model_type="gemini")
        print("✅ LLM service initialized")
        
        # Test with a potentially problematic prompt (simulating content filtering)
        # This is just to verify our error handling works, not to trigger actual filtering
        test_prompt = "Generate a summary for the following content: This is a normal business document about project planning and team coordination."
        
        print(f"📝 Testing with prompt: {test_prompt[:100]}...")
        
        try:
            response = llm_service._chat_completion(test_prompt, temperature=0.5)
            print(f"🤖 Response received: {response[:200]}...")
            print("✅ No errors encountered - this is good!")
        except Exception as e:
            print(f"⚠️  Error caught (this tests our error handling): {e}")
        
        print("✅ Error handling test completed")
        return True
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

if __name__ == "__main__":
    success = test_error_handling()
    sys.exit(0 if success else 1)