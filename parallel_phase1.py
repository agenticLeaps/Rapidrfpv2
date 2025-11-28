#!/usr/bin/env python3
"""
Replacement for async phase 1 with true parallel processing
"""

import concurrent.futures
from typing import Dict, Any
from ..llm.llm_service import ExtractionResult
import logging

logger = logging.getLogger(__name__)

async def parallel_phase_1_decomposition(self, processed_doc) -> Dict[str, Any]:
    """Optimized parallel Phase I with ThreadPoolExecutor."""
    total_chunks = len(processed_doc.chunks)
    print(f"🚀 PARALLEL PROCESSING STARTED: {total_chunks} chunks with 16 workers")
    print(f"⚡ Each chunk will be processed simultaneously using ThreadPoolExecutor")
    
    try:
        def process_single_chunk(chunk_data):
            """Process a single chunk and create nodes."""
            chunk, chunk_index = chunk_data
            try:
                print(f"   🔄 Worker processing chunk {chunk_index+1}/{total_chunks}...")
                
                # Extract using LLM
                result = self.llm_service.extract_all_from_chunk(chunk.content)
                
                if result.success:
                    # Create nodes from result
                    success = self._process_chunk(chunk)  # Use existing method
                    print(f"   ✅ Chunk {chunk_index+1}: {len(result.entities)} entities, {len(result.relationships)} relationships")
                    return (True, len(result.entities), len(result.relationships), len(result.semantic_units))
                else:
                    print(f"   ❌ Chunk {chunk_index+1}: LLM extraction failed - {result.error_message}")
                    return (False, 0, 0, 0)
                    
            except Exception as e:
                print(f"   ❌ Chunk {chunk_index+1}: Exception - {e}")
                return (False, 0, 0, 0)
        
        # Process all chunks in parallel using ThreadPoolExecutor
        print(f"⚡ Starting {total_chunks} parallel extraction tasks with 16 workers...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Submit all chunk processing tasks with index
            chunk_data = [(chunk, i) for i, chunk in enumerate(processed_doc.chunks)]
            futures = [executor.submit(process_single_chunk, data) for data in chunk_data]
            
            successful_chunks = 0
            failed_chunks = 0
            total_entities = 0
            total_relationships = 0
            total_semantic_units = 0
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    success, entities, relationships, semantic_units = future.result()
                    
                    if success:
                        successful_chunks += 1
                        total_entities += entities
                        total_relationships += relationships
                        total_semantic_units += semantic_units
                    else:
                        failed_chunks += 1
                        
                except Exception as e:
                    failed_chunks += 1
                    print(f"   ❌ Future result error: {e}")
        
        print(f"📊 PARALLEL PROCESSING COMPLETED!")
        print(f"   Success: {successful_chunks}/{total_chunks} chunks")
        print(f"   Extracted: {total_entities} entities, {total_relationships} relationships, {total_semantic_units} semantic units")
        
        logger.info(f"Parallel Phase I completed: {successful_chunks} successful, {failed_chunks} failed")
        
        return {
            'success': True,
            'chunks_processed': successful_chunks,
            'chunks_failed': failed_chunks,
            'processing_method': 'true_parallel_threadpool'
        }
        
    except Exception as e:
        print(f"❌ PARALLEL PROCESSING FAILED: {e}")
        logger.error(f"Error in Parallel Phase I: {e}")
        return {'success': False, 'error': str(e)}