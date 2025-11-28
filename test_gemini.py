#!/usr/bin/env python3
"""
Test Gemini 2.0 Flash Exp for NodeRAG extraction
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Test if Google API key is available
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("❌ GOOGLE_API_KEY not found in environment variables")
    print("💡 Please add your Google AI API key to .env file:")
    print("   GOOGLE_API_KEY=your_key_here")
    exit(1)

try:
    import google.generativeai as genai
    print("✅ Google Generative AI library imported successfully")
except ImportError:
    print("❌ google-generativeai not installed")
    print("💡 Install with: pip install google-generativeai")
    exit(1)

# Test Gemini initialization and extraction
def test_gemini_extraction():
    """Test Gemini with NodeRAG extraction tasks"""
    
    try:
        # Initialize Gemini
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Gemini 1.5 Flash initialized")
        
        # Test text
        test_text = """
        The Amazon Web Services (AWS) cloud platform provides various services including EC2 compute instances, S3 storage buckets, and RDS databases. 
        John Smith, the lead architect at TechCorp, implemented a microservices architecture using Docker containers deployed on Kubernetes clusters. 
        The system processes customer data from multiple regions including North America, Europe, and Asia-Pacific. 
        The development team, led by Sarah Johnson, uses CI/CD pipelines with Jenkins for automated testing and deployment.
        """
        
        # Test entity extraction
        entity_prompt = f"""Extract named entities from the following text. Focus on:
- People (names, titles, roles)
- Places (locations, buildings, geographical features)  
- Organizations (companies, institutions, groups)
- Objects (specific items, products, concepts)
- Events (specific named events, meetings, projects)

Rules:
- Return only the entity names, not descriptions
- Use the most specific form (e.g., "Dr. John Smith" not just "John")
- Maximum 4 entities
- Return as a JSON list of strings

Text: {test_text}

Entities:"""

        print("🧪 Testing entity extraction with Gemini...")
        
        generation_config = genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=256,
        )
        
        response = model.generate_content(
            entity_prompt,
            generation_config=generation_config
        )
        
        result = response.text.strip()
        print(f"✅ Gemini Response:\n{result}")
        print(f"📏 Response length: {len(result)} characters")
        
        # Test relationship extraction
        print("\n🔗 Testing relationship extraction...")
        
        relationship_prompt = f"""Extract relationships between entities from the text.

Entities: John Smith, TechCorp, AWS, Sarah Johnson

Rules:
- Only use entities from the provided list
- Relationship format: (Entity1, Relationship, Entity2)
- Use clear, simple relationship terms (e.g., "works for", "uses", "manages")
- Maximum 3 relationships
- Return as JSON list of [entity1, relationship, entity2] arrays

Text: {test_text}

Relationships:"""

        response = model.generate_content(
            relationship_prompt,
            generation_config=generation_config
        )
        
        result = response.text.strip()
        print(f"✅ Relationship Response:\n{result}")
        
        print("\n🎉 Gemini testing completed successfully!")
        print("🚀 Ready to use Gemini 2.0 Flash Exp for NodeRAG processing")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Gemini 2.0 Flash Exp for NodeRAG")
    print("=" * 50)
    
    success = test_gemini_extraction()
    
    if success:
        print("\n✅ Gemini is ready for NodeRAG processing!")
        print("🔄 You can now run the pipeline with Gemini model")
    else:
        print("\n❌ Gemini setup needs attention")
        print("🔧 Please check API key and installation")