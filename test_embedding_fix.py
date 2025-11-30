#!/usr/bin/env python3
"""
Quick test script to verify the embedding fix works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.llm_service import LLMService

def test_embedding_fix():
    print("🧪 Testing embedding fix...")
    
    try:
        # Initialize LLM service with Gemini (which was causing the issue)
        llm_service = LLMService(model_type="gemini")
        print("✅ LLM service initialized")
        
        # Test small embedding batch
        test_texts = [
            "This is a test sentence.",
            "Another test sentence for embeddings.",
            "Third test sentence to verify batch processing."
        ]
        
        print(f"🔍 Testing embeddings for {len(test_texts)} texts...")
        embeddings = llm_service.get_embeddings(test_texts, batch_size=2)
        
        if embeddings and len(embeddings) == len(test_texts):
            print(f"✅ Success! Generated {len(embeddings)} embeddings")
            print(f"📏 Embedding dimension: {len(embeddings[0])}")
            return True
        else:
            print(f"❌ Failed! Expected {len(test_texts)} embeddings, got {len(embeddings) if embeddings else 0}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_embedding_fix()
    sys.exit(0 if success else 1)