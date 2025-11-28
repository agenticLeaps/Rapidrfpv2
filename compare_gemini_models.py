#!/usr/bin/env python3
"""
Gemini Model Comparison Script
Compares gemini-2.5-flash-lite, gemini-2.0-flash-lite-001, and gemini-2.0-flash-001
on chunk processing and saves results to separate files for analysis.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.llm_service import LLMService
import google.generativeai as genai

class GeminiModelComparator:
    def __init__(self):
        self.models = {
            'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite',
            'gemini-2.0-flash-lite-001': 'gemini-2.0-flash-lite-001', 
            'gemini-2.0-flash-001': 'gemini-2.0-flash-001'
        }
        self.results = {}
        
    def process_chunk_with_model(self, model_name: str, chunk_text: str) -> Dict[str, Any]:
        """Process a chunk with a specific Gemini model."""
        print(f"\n{'='*50}")
        print(f"🤖 Testing Model: {model_name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        result = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'chunk_text': chunk_text,
            'entities': [],
            'relationships': [],
            'semantic_units': [],
            'errors': [],
            'safety_blocks': 0,
            'api_calls': 0,
            'tokens': {'input': 0, 'output': 0, 'total': 0},
            'processing_time': 0,
            'success': False
        }
        
        try:
            # Configure Gemini with the specific model
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(self.models[model_name])
            
            # Create a custom LLM service instance for this model
            llm_service = LLMService(model_type="gemini")
            llm_service.gemini_model = model  # Override with specific model
            
            print(f"✅ {model_name} initialized")
            
            # Extract entities
            print("🔍 Extracting entities...")
            try:
                entities = llm_service.extract_entities(chunk_text, max_entities=20)
                result['entities'] = entities
                print(f"   ✅ Extracted {len(entities)} entities")
            except Exception as e:
                error_msg = f"Entity extraction failed: {str(e)}"
                result['errors'].append(error_msg)
                print(f"   ❌ {error_msg}")
                if "finish_reason" in str(e):
                    result['safety_blocks'] += 1
            
            # Extract relationships (only if entities were found)
            if result['entities']:
                print("🔗 Extracting relationships...")
                try:
                    relationships = llm_service.extract_relationships(chunk_text, result['entities'], max_relationships=15)
                    result['relationships'] = relationships
                    print(f"   ✅ Extracted {len(relationships)} relationships")
                except Exception as e:
                    error_msg = f"Relationship extraction failed: {str(e)}"
                    result['errors'].append(error_msg)
                    print(f"   ❌ {error_msg}")
                    if "finish_reason" in str(e):
                        result['safety_blocks'] += 1
            
            # Extract semantic units
            print("📝 Extracting semantic units...")
            try:
                semantic_units = llm_service.extract_semantic_units(chunk_text, max_units=5)
                result['semantic_units'] = semantic_units
                print(f"   ✅ Extracted {len(semantic_units)} semantic units")
            except Exception as e:
                error_msg = f"Semantic unit extraction failed: {str(e)}"
                result['errors'].append(error_msg)
                print(f"   ❌ {error_msg}")
                if "finish_reason" in str(e):
                    result['safety_blocks'] += 1
            
            # Get usage statistics
            stats = llm_service.get_usage_stats()
            result['api_calls'] = stats['api_calls']
            result['tokens'] = {
                'input': stats['input_tokens'],
                'output': stats['output_tokens'], 
                'total': stats['total_tokens']
            }
            
            result['success'] = len(result['errors']) == 0
            
        except Exception as e:
            error_msg = f"Model initialization or processing failed: {str(e)}"
            result['errors'].append(error_msg)
            print(f"❌ {error_msg}")
        
        result['processing_time'] = time.time() - start_time
        
        # Print summary
        print(f"\n📊 {model_name} Summary:")
        print(f"   Entities: {len(result['entities'])}")
        print(f"   Relationships: {len(result['relationships'])}")
        print(f"   Semantic Units: {len(result['semantic_units'])}")
        print(f"   API Calls: {result['api_calls']}")
        print(f"   Total Tokens: {result['tokens']['total']:,}")
        print(f"   Safety Blocks: {result['safety_blocks']}")
        print(f"   Errors: {len(result['errors'])}")
        print(f"   Processing Time: {result['processing_time']:.2f}s")
        print(f"   Success: {'✅' if result['success'] else '❌'}")
        
        return result
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "model_comparison_results"):
        """Save results to individual files for each model."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, result in results.items():
            # Create detailed result file
            filename = f"{output_dir}/{model_name}_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Create human-readable summary
            summary_filename = f"{output_dir}/{model_name}_{timestamp}_summary.txt"
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(f"Gemini Model Comparison Results\n")
                f.write(f"{'='*40}\n\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Timestamp: {result['timestamp']}\n")
                f.write(f"Processing Time: {result['processing_time']:.2f}s\n")
                f.write(f"Success: {'✅' if result['success'] else '❌'}\n\n")
                
                f.write(f"Statistics:\n")
                f.write(f"  - Entities Found: {len(result['entities'])}\n")
                f.write(f"  - Relationships Found: {len(result['relationships'])}\n") 
                f.write(f"  - Semantic Units Found: {len(result['semantic_units'])}\n")
                f.write(f"  - API Calls Made: {result['api_calls']}\n")
                f.write(f"  - Input Tokens: {result['tokens']['input']:,}\n")
                f.write(f"  - Output Tokens: {result['tokens']['output']:,}\n")
                f.write(f"  - Total Tokens: {result['tokens']['total']:,}\n")
                f.write(f"  - Safety Blocks: {result['safety_blocks']}\n")
                f.write(f"  - Errors: {len(result['errors'])}\n\n")
                
                if result['entities']:
                    f.write(f"Entities Extracted:\n")
                    for i, entity in enumerate(result['entities'], 1):
                        f.write(f"  {i}. {entity}\n")
                    f.write(f"\n")
                
                if result['relationships']:
                    f.write(f"Relationships Extracted:\n")
                    for i, rel in enumerate(result['relationships'], 1):
                        f.write(f"  {i}. {rel[0]} → {rel[1]} → {rel[2]}\n")
                    f.write(f"\n")
                
                if result['semantic_units']:
                    f.write(f"Semantic Units Extracted:\n")
                    for i, unit in enumerate(result['semantic_units'], 1):
                        f.write(f"  {i}. {unit}\n")
                    f.write(f"\n")
                
                if result['errors']:
                    f.write(f"Errors Encountered:\n")
                    for i, error in enumerate(result['errors'], 1):
                        f.write(f"  {i}. {error}\n")
                    f.write(f"\n")
                
                f.write(f"Input Chunk Text:\n")
                f.write(f"{'='*20}\n")
                f.write(f"{result['chunk_text']}\n")
        
        # Create comparison summary
        comparison_filename = f"{output_dir}/comparison_summary_{timestamp}.txt"
        with open(comparison_filename, 'w', encoding='utf-8') as f:
            f.write(f"Gemini Models Comparison Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"{'Model':<25} {'Entities':<10} {'Relations':<10} {'Semantic':<10} {'Tokens':<10} {'Blocks':<8} {'Success':<8}\n")
            f.write(f"{'-'*80}\n")
            
            for model_name, result in results.items():
                f.write(f"{model_name:<25} ")
                f.write(f"{len(result['entities']):<10} ")
                f.write(f"{len(result['relationships']):<10} ")
                f.write(f"{len(result['semantic_units']):<10} ")
                f.write(f"{result['tokens']['total']:<10} ")
                f.write(f"{result['safety_blocks']:<8} ")
                f.write(f"{'✅' if result['success'] else '❌':<8}\n")
        
        print(f"\n📁 Results saved to '{output_dir}' directory:")
        print(f"   - Individual model results: {len(results)} JSON files")
        print(f"   - Human-readable summaries: {len(results)} text files")
        print(f"   - Comparison summary: comparison_summary_{timestamp}.txt")
    
    def compare_models(self, chunk_text: str):
        """Compare all three Gemini models on the given chunk."""
        print(f"🚀 Starting Gemini Model Comparison")
        print(f"📄 Chunk length: {len(chunk_text)} characters")
        print(f"📄 Chunk preview: {chunk_text[:100]}...")
        
        results = {}
        
        for model_key, model_name in self.models.items():
            try:
                result = self.process_chunk_with_model(model_key, chunk_text)
                results[model_key] = result
                time.sleep(1)  # Small delay between models to avoid rate limits
            except Exception as e:
                print(f"❌ Failed to test {model_key}: {e}")
        
        # Save all results
        self.save_results(results)
        
        print(f"\n🎉 Model comparison completed!")
        print(f"📊 Tested {len(results)} models successfully")
        
        return results

def main():
    """Main function to run the comparison."""
    print("🧪 Gemini Model Comparison Tool")
    print("=" * 50)
    
    # Sample chunk text for testing (you can modify this)
    default_chunk = """
    Andor Health is delighted to respond to the RFI titled Strategic System Selection - Innovation. We are eager to demonstrate our ability to meet and exceed the requirements as defined in the RFI by Beebe Healthcare.
     
    Andor Health was born 4 years ago with a single mission; to fundamentally change the way in which care teams, patients, and families connect and collaborate. By harnessing the latest innovations in OpenAI/ChatGPT and transformative generative AI/ML models, our cloud-based platform unlocks data stored in source systems, such as electronic medical records, to deliver real-time actionable intelligence to care teams within ubiquitous care team collaboration platforms like Microsoft Teams. By perfecting communication workflows, our platform accelerates time to treatment, decreases clinician burnout, and drives better patient outcomes.
     
    For over 20 years, our leadership team has placed intense focus on pioneering technologies for patients, creating more collaborative experiences for clinicians, using innovative devices, and driving evolution to mobile self-service. Healthcare institutions and care teams use ThinkAndor® to enable providers to configure patient and clinician interactions with ubiquitous team collaboration platforms. This eliminates the need to manage added applications. ThinkAndor® enables a frictionless virtual interaction allowing physicians and patients to communicate without being distracted by disjointed technologies during a virtual consultation. ThinkAndor is the only integrated virtual collaboraiton platform that can truly integrate all aspects of ambulatory, acute, post-acute and at home virtual care collaboration through our 5 Pillars of Virtual Health: Virtual Visits, Virtual Rounding, Virtual Patient Monitoring, Virtual Team Collaboration, and Virtual Community Collaboration.
     
    Andor Health is partnered with over 40 health systems across the US, Uk, and Canada, with over 70,000 providers and 300 hospitals leveraging the Andor Health platform globally, and ThinkAndor is the only virtual collaboration platform deployed AT SCALE. National Institutes of Health, Orlando Health, Tampa General, Medical University of South Carolina, and NHS, are among some of the most notable. You may find a list of our strategic partners by accessing this link. Moreover we have several distinct strategic partnerships and integrations, we have integrated to multiple EMR's including being a top tier partner Epic Connection Hub (App Orchard) partner, Cerner (CODE development partner), and Athena MDP partner which allow ThinkAndor to discretely integrate data and workflows to remain fundamentally aligned with any back end data sources at Beebe and retain consistent workflows for care teams.
    """
    
    # Check if chunk is provided as command line argument
    if len(sys.argv) > 1:
        chunk_text = sys.argv[1]
        print("📝 Using provided chunk text")
    else:
        chunk_text = default_chunk.strip()
        print("📝 Using default sample chunk text")
        print("💡 To use custom text: python compare_gemini_models.py 'Your text here'")
    
    # Initialize comparator
    comparator = GeminiModelComparator()
    
    # Run comparison
    try:
        results = comparator.compare_models(chunk_text)
        
        print("\n" + "=" * 50)
        print("🏁 FINAL COMPARISON SUMMARY")
        print("=" * 50)
        
        for model_name, result in results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"{model_name:25} | {status} | Entities: {len(result['entities']):2} | Relations: {len(result['relationships']):2} | Tokens: {result['tokens']['total']:4} | Blocks: {result['safety_blocks']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")

if __name__ == "__main__":
    main()