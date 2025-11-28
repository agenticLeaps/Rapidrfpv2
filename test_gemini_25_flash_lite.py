#!/usr/bin/env python3
"""
Test script to verify Gemini 2.5 Flash Lite model functionality.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.llm_service import LLMService

def test_gemini_25_flash_lite():
    """Test basic functionality with Gemini 2.5 Flash Lite."""
    print("🧪 Testing Gemini 2.5 Flash Lite model...")
    
    try:
        # Initialize LLM service with Gemini
        llm_service = LLMService(model_type="gemini")
        print("✅ LLM service initialized")
        
        # Test basic text completion
        test_prompt = "What is the capital of France?"
        response = llm_service._chat_completion(test_prompt, temperature=0.3, max_tokens=100)
        
        print(f"📝 Test prompt: {test_prompt}")
        print(f"🤖 Model response: {response}")
        
        if response and "Paris" in response:
            print("✅ Basic completion test passed")
        else:
            print("❌ Basic completion test failed")
            return False
            
        # Test entity extraction
        test_text = "John Smith works at Google in Mountain View, California."
        entities = llm_service.extract_entities(test_text)
        
        print(f"📝 Test text: {test_text}")
        print(f"🏷️ Extracted entities: {entities}")
        
        if entities and any("John" in entity for entity in entities):
            print("✅ Entity extraction test passed")
        else:
            print("❌ Entity extraction test failed")
            return False
            
        # Print usage statistics
        llm_service.print_usage_summary()
        
        print("🎉 All tests passed! Gemini 2.5 Flash Lite is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_gemini_25_flash_lite()
    sys.exit(0 if success else 1)