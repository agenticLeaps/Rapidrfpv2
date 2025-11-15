#!/usr/bin/env python3
"""
Test script to debug LLM endpoints
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gradio_client import Client
from src.config.settings import Config

def test_llm_endpoints():
    print("Testing LLM endpoints...")
    
    try:
        print(f"Connecting to LLM endpoint: {Config.QWEN_LLM_ENDPOINT}")
        llm_client = Client(Config.QWEN_LLM_ENDPOINT)
        
        print("LLM client created successfully")
        
        # View available API endpoints
        try:
            api_info = llm_client.view_api()
            print(f"LLM API Info: {api_info}")
        except Exception as e:
            print(f"Could not view API: {e}")
        
        # Try different API names
        test_prompts = ["Hello, world!"]
        api_names_to_try = ["/predict", "/generate", "/chat", None]
        
        for api_name in api_names_to_try:
            try:
                print(f"\nTrying API name: {api_name}")
                if api_name:
                    result = llm_client.predict("What is 2+2?", api_name=api_name)
                else:
                    result = llm_client.predict("What is 2+2?")
                print(f"Success with {api_name}: {result}")
                break
            except Exception as e:
                print(f"Failed with {api_name}: {e}")
        
    except Exception as e:
        print(f"Failed to connect to LLM: {e}")
    
    try:
        print(f"\nConnecting to embedding endpoint: {Config.QWEN_EMBEDDING_ENDPOINT}")
        embedding_client = Client(Config.QWEN_EMBEDDING_ENDPOINT)
        
        print("Embedding client created successfully")
        
        # View available API endpoints
        try:
            api_info = embedding_client.view_api()
            print(f"Embedding API Info: {api_info}")
        except Exception as e:
            print(f"Could not view embedding API: {e}")
            
        # Test embedding
        try:
            result = embedding_client.predict("Hello world", 1, api_name="/predict")
            print(f"Embedding test successful: {type(result)}")
        except Exception as e:
            print(f"Embedding test failed: {e}")
        
    except Exception as e:
        print(f"Failed to connect to embedding service: {e}")

if __name__ == "__main__":
    test_llm_endpoints()