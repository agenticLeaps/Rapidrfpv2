#!/usr/bin/env python3
"""
Test script to demonstrate the performance logging system.
This script simulates a file upload and processing to show how the logging works.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.performance_logger import performance_logger
from src.config.settings import Config

def create_test_document(content: str) -> str:
    """Create a temporary test document."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return f.name

def simulate_file_processing():
    """Simulate the file processing pipeline with performance logging."""
    
    print("🚀 Starting Performance Logging Demo")
    print("=" * 60)
    
    # Create a test document
    test_content = """
    This is a sample document for testing the performance logging system.
    
    The document contains information about artificial intelligence, machine learning,
    and natural language processing. These technologies are transforming various
    industries including healthcare, finance, and education.
    
    Key concepts include:
    - Neural networks and deep learning
    - Natural language understanding
    - Computer vision and image recognition
    - Automated decision making
    
    Companies like OpenAI, Google, and Microsoft are leading research in this field.
    The applications range from chatbots and virtual assistants to autonomous
    vehicles and medical diagnosis systems.
    """
    
    test_file = create_test_document(test_content)
    file_size = os.path.getsize(test_file)
    
    try:
        # Start a processing session
        session_id = "demo_session_" + str(int(time.time()))
        performance_logger.start_session(session_id, "test_document.txt", file_size)
        
        print(f"📄 Processing document: {os.path.basename(test_file)} ({file_size} bytes)")
        print(f"🔖 Session ID: {session_id}")
        print()
        
        # Simulate document loading
        with performance_logger.step("Document Loading", file_path=test_file):
            print("   📂 Loading document...")
            time.sleep(0.5)  # Simulate loading time
            performance_logger.add_step_metadata(
                chunks_created=3,
                parsing_method="text_loader"
            )
        
        # Simulate graph decomposition
        with performance_logger.step("Graph Decomposition", total_chunks=3):
            print("   🔗 Decomposing into graph nodes...")
            
            # Simulate chunk processing
            for i in range(3):
                with performance_logger.step(f"Process Chunk {i+1}", chunk_index=i):
                    print(f"      ⚡ Processing chunk {i+1}/3...")
                    time.sleep(0.3)  # Simulate processing time
                    
                    # Simulate sub-steps
                    with performance_logger.step("Create Text Node"):
                        time.sleep(0.1)
                        performance_logger.add_step_metadata(text_node_id=f"T_chunk_{i}")
                    
                    with performance_logger.step("LLM Extraction"):
                        time.sleep(0.4)
                        performance_logger.add_step_metadata(
                            entities_found=2 + i,
                            relationships_found=1 + i,
                            semantic_units_found=3
                        )
                    
                    with performance_logger.step("Create Entity Nodes"):
                        time.sleep(0.2)
                        performance_logger.add_step_metadata(entity_nodes_created=2 + i)
                    
                    with performance_logger.step("Create Relationship Nodes"):
                        time.sleep(0.1)
                        performance_logger.add_step_metadata(relationships_created=1 + i)
        
        # Simulate graph augmentation
        with performance_logger.step("Graph Augmentation"):
            print("   🎯 Augmenting graph with important entities...")
            
            with performance_logger.step("Identify Important Entities"):
                time.sleep(0.2)
                performance_logger.add_step_metadata(important_entities_count=5)
            
            with performance_logger.step("Generate Attribute Nodes"):
                time.sleep(0.8)
                performance_logger.add_step_metadata(attribute_nodes_created=5)
            
            with performance_logger.step("Community Detection"):
                time.sleep(0.4)
                performance_logger.add_step_metadata(communities_detected=2)
            
            with performance_logger.step("Generate Community Summaries"):
                time.sleep(0.6)
                performance_logger.add_step_metadata(
                    high_level_nodes_created=2,
                    overview_nodes_created=2
                )
        
        # Simulate embedding generation
        with performance_logger.step("Embedding Generation & Storage"):
            print("   🧠 Generating embeddings...")
            
            with performance_logger.step("Initialize HNSW Service"):
                time.sleep(0.1)
                performance_logger.add_step_metadata(hnsw_initialized=True)
            
            with performance_logger.step("Collect Nodes for Embedding"):
                time.sleep(0.1)
                total_nodes = 25
                performance_logger.add_step_metadata(
                    total_nodes=total_nodes,
                    node_type_breakdown={
                        "ENTITY": 8,
                        "SEMANTIC": 9,
                        "RELATIONSHIP": 4,
                        "ATTRIBUTE": 2,
                        "HIGH_LEVEL": 1,
                        "OVERVIEW": 1
                    }
                )
            
            with performance_logger.step("Generate Embeddings", total_batches=3):
                # Simulate batch processing
                for batch in range(3):
                    with performance_logger.step(f"Process Batch {batch+1}"):
                        print(f"      🔢 Processing embedding batch {batch+1}/3...")
                        time.sleep(0.5)
                        performance_logger.add_step_metadata(
                            embeddings_in_batch=8 if batch < 2 else 9,
                            hnsw_indexed_in_batch=8 if batch < 2 else 9
                        )
            
            with performance_logger.step("Save HNSW Index"):
                time.sleep(0.2)
                performance_logger.add_step_metadata(
                    save_success=True,
                    index_size=total_nodes
                )
        
        # Simulate graph storage
        with performance_logger.step("Graph Storage"):
            print("   💾 Saving graph to storage...")
            time.sleep(0.3)
            performance_logger.add_step_metadata(graph_saved=True)
        
        # End the session
        completed_session = performance_logger.end_session('completed')
        
        print()
        print("✅ Processing completed successfully!")
        print(f"⏱️  Total Duration: {completed_session.total_duration_formatted}")
        print()
        
        # Generate and display the report
        print("📊 PERFORMANCE REPORT")
        print("=" * 60)
        
        report = performance_logger.get_session_report(session_id)
        
        print(f"Session ID: {report['session_id']}")
        print(f"File: {report['file_info']['name']}")
        print(f"Size: {report['file_info']['size_formatted']}")
        print(f"Duration: {report['timing']['total_duration_formatted']}")
        print(f"Status: {report['status']}")
        
        if report['summary']:
            summary = report['summary']
            print(f"Processing Rate: {summary['processing_rate']:.1f} bytes/sec")
            print(f"Steps: {summary['successful_steps']}/{summary['total_steps']} successful")
            print()
        
        print("Step Breakdown:")
        print("-" * 40)
        for step in report['steps']:
            status_symbol = "✓" if step['status'] == 'completed' else "✗"
            print(f"{status_symbol} {step['name']:<25} {step['duration_formatted']:>10}")
            if step['sub_steps_count'] > 0:
                print(f"  └─ Sub-steps: {step['sub_steps_count']}")
        
        # Export report to file
        print("\n📄 Exporting detailed report...")
        report_path = performance_logger.export_session_report(session_id)
        print(f"Report saved to: {report_path}")
        
        return session_id, report_path
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        performance_logger.end_session('failed', str(e))
        return None, None
        
    finally:
        # Clean up test file
        try:
            os.unlink(test_file)
        except:
            pass

def main():
    """Main function to run the demo."""
    print("Performance Logging System Demo")
    print("This script demonstrates the comprehensive logging system for file processing")
    print()
    
    # Set up data directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "performance_logs"), exist_ok=True)
    
    # Run the simulation
    session_id, report_path = simulate_file_processing()
    
    if session_id:
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Log files are available in: {performance_logger.log_dir}")
        print(f"📊 Detailed report: {report_path}")
        print("\nNext steps:")
        print("1. Upload a file using the API to see real performance logging")
        print("2. Use the /api/performance/report/<session_id> endpoint to get reports")
        print("3. Monitor processing with /api/performance/current endpoint")
        print("4. Stream logs with /api/performance/logs/stream endpoint")
    else:
        print("\n❌ Demo failed. Check the error messages above.")

if __name__ == "__main__":
    main()