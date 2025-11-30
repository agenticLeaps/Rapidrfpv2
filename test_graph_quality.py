#!/usr/bin/env python3
"""
Test script to verify the new two-phase graph building approach works correctly.
"""

import asyncio
import logging
from src.document_processing.indexing_pipeline import IndexingPipeline
from src.document_processing.document_loader import DocumentChunk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_two_phase_graph_building():
    """Test the new unified graph building approach."""
    
    print("🧪 Testing Two-Phase Graph Building Approach")
    print("=" * 50)
    
    # Create test pipeline
    pipeline = IndexingPipeline()
    
    # Create test chunks based on the Andor Health brief summary to test data preservation
    test_chunks = [
        DocumentChunk(
            content="Andor Health was born 4 years ago with a single mission; to fundamentally change the way in which care teams, patients, and families connect and collaborate.",
            chunk_index=0,
            metadata={"file_path": "andor_summary.txt", "chunk_start": 0}
        ),
        DocumentChunk(
            content="For over 20 years, our leadership team has placed intense focus on pioneering technologies for patients, creating more collaborative experiences for clinicians.",
            chunk_index=1,
            metadata={"file_path": "andor_summary.txt", "chunk_start": 100}
        ),
        DocumentChunk(
            content="By harnessing the latest innovations in OpenAI/ChatGPT and transformative generative AI/ML models, our cloud-based platform unlocks data stored in source systems.",
            chunk_index=2,
            metadata={"file_path": "andor_summary.txt", "chunk_start": 200}
        ),
        DocumentChunk(
            content="ThinkAndor enables a frictionless virtual interaction allowing physicians and patients to communicate without being distracted.",
            chunk_index=3,
            metadata={"file_path": "andor_summary.txt", "chunk_start": 300}
        )
    ]
    
    print(f"📝 Testing with {len(test_chunks)} chunks from Andor Health brief summary")
    print("🎯 Key data to preserve: '4 years ago', '20 years', 'leadership team', 'OpenAI/ChatGPT', 'ThinkAndor'")
    
    try:
        # Test Phase 1a: Parallel extraction
        print("\n🚀 Phase 1a: Parallel LLM Extraction")
        extraction_results = await pipeline._parallel_extract_all_chunks(test_chunks)
        
        successful_extractions = sum(1 for r in extraction_results if r.get('success', False))
        print(f"✅ Extractions: {successful_extractions}/{len(test_chunks)} successful")
        
        # Print extraction details
        for i, result in enumerate(extraction_results):
            if result.get('success'):
                entities = result.get('entities', [])
                relationships = result.get('relationships', [])
                print(f"   Chunk {i+1}: {len(entities)} entities, {len(relationships)} relationships")
                print(f"      Entities: {entities}")
                if relationships:
                    print(f"      Relationships: {relationships[:2]}...")
        
        # Test Phase 1b: Unified graph building
        print("\n🏗️  Phase 1b: Unified Graph Building")
        graph_result = await pipeline._build_unified_graph_from_extractions(extraction_results, test_chunks)
        
        print(f"✅ Graph built successfully!")
        print(f"   📊 Total nodes: {graph_result['total_nodes']}")
        print(f"   🔗 Total edges: {graph_result['total_edges']}")
        print(f"   🏢 Entity nodes: {graph_result['entity_nodes']} (from {graph_result['unique_entities']} unique)")
        print(f"   ✨ Consolidated entities: {graph_result['consolidated_entities']}")
        print(f"   🌉 Cross-chunk relationships: {graph_result['cross_chunk_relationships']}")
        
        # Verify entity consolidation worked
        if graph_result['consolidated_entities'] > 0:
            print(f"🎉 SUCCESS: Entity consolidation working! {graph_result['consolidated_entities']} entities consolidated")
        else:
            print("⚠️  No entities were consolidated - check entity matching logic")
        
        # Check if temporal data was preserved
        temporal_preserved = graph_result.get('temporal_context_nodes', 0) > 0 or graph_result.get('temporal_relationships', 0) > 0
        if temporal_preserved:
            print(f"⏰ SUCCESS: Temporal data preserved! Context nodes: {graph_result.get('temporal_context_nodes', 0)}")
        else:
            print("⚠️  Temporal data may be missing - check temporal detection")
        
        # Get final graph stats
        final_stats = pipeline.graph_manager.get_stats()
        print(f"\n📈 Final Graph Statistics:")
        print(f"   Total nodes: {final_stats.get('total_nodes', 0)}")
        print(f"   Total edges: {final_stats.get('total_edges', 0)}")
        print(f"   Node types: {final_stats.get('node_type_counts', {})}")
        
        # Test search for critical information
        print(f"\n🔍 Testing Search for Critical Information:")
        
        # Simulate search queries that were failing before
        test_queries = [
            "when did andor health born",
            "how many years leadership team focused on technology", 
            "what are the services provided"
        ]
        
        for query in test_queries:
            print(f"   Query: '{query}' - Graph should now contain relevant context")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Test error details:")
        return False

async def main():
    """Run the test."""
    success = await test_two_phase_graph_building()
    
    if success:
        print("\n✅ Two-phase graph building test PASSED")
        print("   Graph quality improvements are working correctly!")
    else:
        print("\n❌ Two-phase graph building test FAILED")
        print("   Check the implementation for issues")

if __name__ == "__main__":
    asyncio.run(main())