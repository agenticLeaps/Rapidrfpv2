#!/usr/bin/env python3
"""
Force reload test to bypass Python module caching.
"""

import sys
import os
import importlib

# Clear any cached modules
modules_to_clear = [key for key in sys.modules.keys() if 'src.config' in key or 'config' in key]
for module in modules_to_clear:
    if module in sys.modules:
        del sys.modules[module]

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_with_fresh_imports():
    print("🔄 Force Reload Test - Bypassing Module Cache")
    print("=" * 50)
    
    # Step 1: Test dotenv loading
    print("🔍 Step 1: Fresh dotenv load")
    print("-" * 30)
    
    from dotenv import load_dotenv
    
    # Force reload .env
    load_dotenv(override=True)
    
    env_key = os.environ.get('OPENAI_API_KEY')
    if env_key:
        masked = env_key[:8] + "..." + env_key[-4:]
        print(f"✅ Environment key after force reload: {masked}")
    else:
        print("❌ No key in environment after reload")
        return False
    
    # Step 2: Fresh import of Config
    print("\n🔍 Step 2: Fresh Config import")
    print("-" * 30)
    
    # Import config with fresh reload
    import src.config.settings as settings_module
    importlib.reload(settings_module)
    
    config_key = settings_module.Config.OPENAI_API_KEY
    if config_key:
        masked_config = config_key[:8] + "..." + config_key[-4:]
        print(f"✅ Config key after fresh import: {masked_config}")
        
        # Check if they match
        if config_key == env_key:
            print("✅ Config key matches environment key!")
            return True
        else:
            print("❌ Config key does NOT match environment key")
            print(f"   Env:    ...{env_key[-4:]}")
            print(f"   Config: ...{config_key[-4:]}")
            return False
    else:
        print("❌ Config key is empty after fresh import")
        return False

def test_openai_with_fresh_key():
    """Test OpenAI with the freshly loaded key."""
    print("\n🔍 Step 3: OpenAI test with fresh key")
    print("-" * 30)
    
    try:
        # Get the fresh key directly from environment
        fresh_key = os.environ.get('OPENAI_API_KEY')
        
        if not fresh_key:
            print("❌ No fresh key available")
            return False
        
        from openai import OpenAI
        
        # Use the fresh key directly
        client = OpenAI(api_key=fresh_key)
        
        print("📡 Testing API call with fresh key...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say exactly: FRESH API KEY WORKS"}],
            max_tokens=10
        )
        
        response_text = response.choices[0].message.content
        print(f"✅ API call successful!")
        print(f"📝 Response: {response_text}")
        return True
        
    except Exception as e:
        print(f"❌ Fresh key test failed: {e}")
        return False

def main():
    """Run the force reload test."""
    # Clear Python cache
    importlib.invalidate_caches()
    
    # Test with fresh imports
    config_works = test_with_fresh_imports()
    
    if config_works:
        # Test OpenAI
        openai_works = test_openai_with_fresh_key()
        
        print("\n📊 FRESH RELOAD SUMMARY")
        print("=" * 30)
        print(f"✅ Config loads correct key: {'Yes' if config_works else 'No'}")
        print(f"✅ OpenAI works with fresh key: {'Yes' if openai_works else 'No'}")
        
        if openai_works:
            print("\n🎉 SUCCESS: Fresh reload fixed the issue!")
            print("Your API key is working. The problem was module caching.")
            print("\n🔧 SOLUTION: Restart any running services:")
            print("   1. Stop api_service.py (Ctrl+C)")
            print("   2. Restart: python api_service.py")
            print("   3. Test: python test_v1_performance_logging.py")
        else:
            print("\n❌ The API key itself might be invalid")
            print("Please check your OpenAI dashboard and verify the key.")
    else:
        print("\n❌ Config import still has issues")
        print("There might be another source of the old key in your code.")

if __name__ == "__main__":
    main()