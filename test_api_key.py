#!/usr/bin/env python3
"""
Quick test to verify the OpenAI API key is working.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import Config
from src.llm.llm_service import LLMService

def test_api_key():
    """Test if the OpenAI API key is working."""
    print("🧪 Testing OpenAI API Key Configuration")
    print("=" * 50)
    
    # Check if API key is loaded
    api_key = Config.OPENAI_API_KEY
    if not api_key:
        print("❌ No OpenAI API key found in configuration")
        return False
    
    # Show key format (masked)
    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"✅ API Key loaded: {masked_key}")
    
    # Test LLM service initialization
    try:
        print("🔧 Initializing LLM service...")
        llm_service = LLMService()
        print("✅ LLM service initialized successfully")
        
        # Test a simple API call
        print("📡 Testing OpenAI API call...")
        test_prompt = "Say 'Hello, I am working!' in exactly those words."
        
        response = llm_service._chat_completion(test_prompt, max_tokens=50)
        
        if response and "working" in response.lower():
            print("✅ OpenAI API key is working correctly!")
            print(f"📝 Response: {response}")
            return True
        else:
            print("⚠️ API call succeeded but response seems unexpected")
            print(f"📝 Response: {response}")
            return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        
        # Check for specific error patterns
        error_str = str(e)
        if "invalid_api_key" in error_str or "401" in error_str:
            print("🔧 Fix: The API key appears to be invalid")
            print("   1. Check your OpenAI API key at https://platform.openai.com/account/api-keys")
            print("   2. Make sure it starts with 'sk-' and has no extra spaces")
            print("   3. Update the .env file with the correct key")
            print("   4. Restart the service")
        elif "billing" in error_str.lower() or "quota" in error_str.lower():
            print("💳 Fix: Check your OpenAI billing/quota")
        else:
            print("🔧 Other API issue - check OpenAI status or your network")
        
        return False

if __name__ == "__main__":
    success = test_api_key()
    if success:
        print("\n🎉 API key test passed! You can now run the performance logging test:")
        print("   python test_v1_performance_logging.py")
    else:
        print("\n❌ API key test failed. Fix the issues above and try again.")