#!/usr/bin/env python3
"""
Debug script to check OpenAI API key configuration step by step.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_env_file():
    """Check .env file directly."""
    print("🔍 Step 1: Checking .env file")
    print("-" * 30)
    
    env_path = ".env"
    if not os.path.exists(env_path):
        print("❌ .env file not found")
        return None
    
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Look for OPENAI_API_KEY
    for line in content.split('\n'):
        if line.strip().startswith('OPENAI_API_KEY='):
            key_value = line.strip().split('=', 1)[1]
            if key_value:
                masked_key = key_value[:8] + "..." + key_value[-4:] if len(key_value) > 12 else "***"
                print(f"✅ Found in .env: {masked_key}")
                print(f"✅ Key length: {len(key_value)} characters")
                print(f"✅ Starts with: {key_value[:7]}...")
                return key_value
            else:
                print("❌ OPENAI_API_KEY found but empty")
                return None
    
    print("❌ OPENAI_API_KEY not found in .env file")
    return None

def check_os_environ():
    """Check if the key is in os.environ."""
    print("\n🔍 Step 2: Checking os.environ")
    print("-" * 30)
    
    key_from_env = os.environ.get('OPENAI_API_KEY')
    if key_from_env:
        masked_key = key_from_env[:8] + "..." + key_from_env[-4:] if len(key_from_env) > 12 else "***"
        print(f"✅ Found in os.environ: {masked_key}")
        return key_from_env
    else:
        print("❌ Not found in os.environ")
        return None

def check_dotenv_loading():
    """Check if dotenv loads correctly."""
    print("\n🔍 Step 3: Testing dotenv loading")
    print("-" * 30)
    
    try:
        from dotenv import load_dotenv
        
        # Load .env manually
        result = load_dotenv()
        print(f"✅ dotenv.load_dotenv() returned: {result}")
        
        # Check after loading
        key_after_load = os.environ.get('OPENAI_API_KEY')
        if key_after_load:
            masked_key = key_after_load[:8] + "..." + key_after_load[-4:] if len(key_after_load) > 12 else "***"
            print(f"✅ Key available after load_dotenv(): {masked_key}")
            return key_after_load
        else:
            print("❌ Key not available after load_dotenv()")
            return None
            
    except ImportError:
        print("❌ python-dotenv not installed")
        return None

def check_config_class():
    """Check if Config class loads the key correctly."""
    print("\n🔍 Step 4: Checking Config class")
    print("-" * 30)
    
    try:
        from src.config.settings import Config
        
        api_key = Config.OPENAI_API_KEY
        if api_key:
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            print(f"✅ Config.OPENAI_API_KEY: {masked_key}")
            print(f"✅ Key length: {len(api_key)} characters")
            return api_key
        else:
            print("❌ Config.OPENAI_API_KEY is empty")
            return None
            
    except Exception as e:
        print(f"❌ Error importing Config: {e}")
        return None

def test_openai_client():
    """Test OpenAI client initialization."""
    print("\n🔍 Step 5: Testing OpenAI client")
    print("-" * 30)
    
    try:
        from src.config.settings import Config
        from openai import OpenAI
        
        print(f"📦 OpenAI library version: {OpenAI.__version__ if hasattr(OpenAI, '__version__') else 'Unknown'}")
        
        # Test client creation
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        print("✅ OpenAI client created successfully")
        
        # Test a simple API call
        print("📡 Testing API call...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say exactly: API KEY WORKS"}],
            max_tokens=10
        )
        
        response_text = response.choices[0].message.content
        print(f"✅ API call successful!")
        print(f"📝 Response: {response_text}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI client test failed: {e}")
        
        error_str = str(e)
        if "invalid_api_key" in error_str or "401" in error_str:
            print("🔧 Diagnosis: Invalid API key")
            print("   - The key format might be wrong")
            print("   - The key might be expired or revoked")
            print("   - Check https://platform.openai.com/account/api-keys")
        elif "billing" in error_str.lower() or "quota" in error_str.lower():
            print("🔧 Diagnosis: Billing/quota issue")
            print("   - Check your OpenAI billing at https://platform.openai.com/account/billing")
        elif "connection" in error_str.lower() or "timeout" in error_str.lower():
            print("🔧 Diagnosis: Network issue")
        else:
            print("🔧 Diagnosis: Other API issue")
        
        return False

def main():
    """Run all diagnostic checks."""
    print("🔧 OpenAI API Key Diagnostic Tool")
    print("=" * 50)
    
    # Step 1: Check .env file
    env_key = check_env_file()
    
    # Step 2: Check os.environ
    environ_key = check_os_environ()
    
    # Step 3: Test dotenv loading
    dotenv_key = check_dotenv_loading()
    
    # Step 4: Check Config class
    config_key = check_config_class()
    
    # Step 5: Test OpenAI client
    client_works = test_openai_client()
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 30)
    print(f"✅ .env file has key: {'Yes' if env_key else 'No'}")
    print(f"✅ os.environ has key: {'Yes' if environ_key else 'No'}")
    print(f"✅ dotenv loads key: {'Yes' if dotenv_key else 'No'}")
    print(f"✅ Config class has key: {'Yes' if config_key else 'No'}")
    print(f"✅ OpenAI client works: {'Yes' if client_works else 'No'}")
    
    if client_works:
        print("\n🎉 SUCCESS: OpenAI API key is working correctly!")
        print("You can now run your performance logging tests.")
    else:
        print("\n❌ ISSUE: OpenAI API key is not working properly.")
        print("Fix the issues above and try again.")
        
        # Specific recommendations
        if env_key and not environ_key:
            print("\n🔧 RECOMMENDATION: Restart your Python process/service")
            print("   The .env file has the key but it's not loaded into the environment")
        elif not env_key:
            print("\n🔧 RECOMMENDATION: Check your .env file")
            print("   Make sure OPENAI_API_KEY=your_actual_key_here")
        elif config_key != env_key:
            print("\n🔧 RECOMMENDATION: Check Config class")
            print("   The Config class is not loading the key from .env properly")

if __name__ == "__main__":
    main()