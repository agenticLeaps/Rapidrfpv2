#!/usr/bin/env python3
"""
Reset embeddings and HNSW index for a fresh start.
Use this when you encounter duplicate node issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.document_processing.indexing_pipeline import IndexingPipeline

def reset_embeddings():
    print("🔄 Resetting embeddings and HNSW index...")
    
    try:
        # Initialize the pipeline
        pipeline = IndexingPipeline()
        
        # Reset embeddings and index
        success = pipeline.reset_embeddings_and_index()
        
        if success:
            print("✅ Embeddings and HNSW index reset successfully!")
            print("You can now run your embedding generation again.")
            return True
        else:
            print("❌ Failed to reset embeddings and index.")
            return False
            
    except Exception as e:
        print(f"❌ Error during reset: {e}")
        return False

if __name__ == "__main__":
    success = reset_embeddings()
    sys.exit(0 if success else 1)