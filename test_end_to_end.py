#!/usr/bin/env python3
"""
End-to-End Test Script for RapidRFP RAG System

This script tests the complete flow:
1. Document processing and indexing
2. Graph construction with all node types
3. Advanced search functionality  
4. Graph visualization generation
5. API endpoint testing

Usage:
    python test_end_to_end.py [--api-url http://localhost:5000]
"""

import os
import sys
import json
import time
import argparse
import tempfile
import requests
from pathlib import Path
from typing import Dict, List, Any

# Test configuration
TEST_API_URL = "http://localhost:5000"
TEST_DOCUMENTS = [
    {
        "content": """# AI and Machine Learning Report

## Introduction
Artificial Intelligence (AI) and Machine Learning (ML) are rapidly evolving fields that are transforming industries worldwide. Companies like Google, Microsoft, and OpenAI are leading the development of advanced AI systems.

## Key Technologies
### Deep Learning
Deep learning uses neural networks with multiple layers to process and analyze data. This technology has enabled breakthroughs in:
- Computer vision
- Natural language processing
- Speech recognition

### Large Language Models
Large Language Models (LLMs) such as GPT-4 and Claude have revolutionized how we interact with AI systems. These models can understand and generate human-like text.

## Industry Applications
AI is being applied across various sectors:
- Healthcare: Medical diagnosis and drug discovery
- Finance: Fraud detection and algorithmic trading
- Transportation: Autonomous vehicles and traffic optimization
- Education: Personalized learning and automated tutoring

## Conclusion
The future of AI holds immense promise, with continued research and development expected to yield even more sophisticated and capable systems.
""",
        "filename": "ai_report.md"
    },
    {
        "content": """# Research Collaboration Agreement

**Parties:** Stanford University and MIT Research Labs
**Project:** Advanced Neural Networks for Natural Language Processing
**Duration:** January 2024 - December 2026

## Project Overview
This collaboration focuses on developing next-generation neural network architectures for natural language understanding and generation.

## Key Researchers
- Dr. Sarah Chen (Stanford University) - Lead AI Researcher
- Dr. Michael Rodriguez (MIT) - Machine Learning Specialist  
- Prof. Lisa Wang (Stanford) - Neural Networks Expert
- Dr. James Thompson (MIT) - Computational Linguistics

## Research Areas
1. **Transformer Architectures**: Improving attention mechanisms
2. **Multimodal Learning**: Combining text, image, and audio processing
3. **Few-shot Learning**: Enabling models to learn from limited data
4. **Model Interpretability**: Understanding how neural networks make decisions

## Deliverables
- 3 research papers per year
- Open-source software implementations
- Graduate student exchange program
- Joint patent applications

## Funding
Total budget: $2.5 million
- Stanford contribution: 60%
- MIT contribution: 40%
- Equipment and infrastructure: $500,000
- Personnel costs: $1.5 million
- Travel and conferences: $300,000
- Publications and dissemination: $200,000

## Timeline
- Phase 1 (Months 1-12): Foundation research and model development
- Phase 2 (Months 13-24): Implementation and testing
- Phase 3 (Months 25-36): Evaluation and publication
""",
        "filename": "collaboration_agreement.md"
    }
]

class EndToEndTester:
    """Comprehensive end-to-end testing suite."""
    
    def __init__(self, api_url: str = TEST_API_URL):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        self.temp_files = []
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details
        })
    
    def make_request(self, method: str, endpoint: str, data: Dict = None, files: Dict = None) -> Dict:
        """Make HTTP request to API."""
        try:
            url = f"{self.api_url}{endpoint}"
            
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=300)
            elif method.upper() == 'POST':
                if files:
                    response = self.session.post(url, data=data, files=files, timeout=300)
                else:
                    response = self.session.post(url, json=data, timeout=300)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'error': f'HTTP {response.status_code}',
                    'details': response.text
                }
        except Exception as e:
            return {
                'error': str(e),
                'details': "Request failed"
            }
    
    def create_test_documents(self) -> List[str]:
        """Create temporary test documents."""
        file_paths = []
        
        for doc in TEST_DOCUMENTS:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=f'.{doc["filename"].split(".")[-1]}',
                delete=False
            ) as f:
                f.write(doc['content'])
                file_path = f.name
            
            self.temp_files.append(file_path)
            file_paths.append(file_path)
        
        return file_paths
    
    def cleanup(self):
        """Clean up temporary files."""
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except:
                pass
    
    def test_api_health(self) -> bool:
        """Test API health endpoint."""
        result = self.make_request('GET', '/health')
        
        if 'error' not in result and result.get('status') == 'healthy':
            self.log_test("API Health Check", True, f"Status: {result.get('status')}")
            return True
        else:
            self.log_test("API Health Check", False, f"Error: {result.get('error', 'Unknown')}")
            return False
    
    def test_document_processing(self, file_paths: List[str]) -> bool:
        """Test document processing and indexing."""
        all_success = True
        
        for i, file_path in enumerate(file_paths):
            # Test file path indexing
            result = self.make_request('POST', '/api/index/document', {
                'file_path': file_path
            })
            
            if 'error' not in result and result.get('success'):
                processing_time = result.get('processing_time', 0)
                chunks_processed = result.get('chunks_processed', 0)
                self.log_test(
                    f"Document {i+1} Indexing", 
                    True, 
                    f"Processed {chunks_processed} chunks in {processing_time:.2f}s"
                )
            else:
                self.log_test(
                    f"Document {i+1} Indexing", 
                    False, 
                    f"Error: {result.get('error', 'Unknown')}"
                )
                all_success = False
        
        return all_success
    
    def test_graph_operations(self) -> bool:
        """Test graph operations and statistics."""
        all_success = True
        
        # Test graph statistics
        result = self.make_request('GET', '/api/graph/stats')
        
        if 'error' not in result:
            stats = result
            total_nodes = stats.get('total_nodes', 0)
            total_edges = stats.get('total_edges', 0)
            communities = stats.get('num_communities', 0)
            
            self.log_test(
                "Graph Statistics", 
                True, 
                f"Nodes: {total_nodes}, Edges: {total_edges}, Communities: {communities}"
            )
            
            # Check if we have nodes of different types
            node_counts = stats.get('node_type_counts', {})
            expected_types = ['T', 'S', 'N', 'R']  # Basic types that should exist
            
            for node_type in expected_types:
                if node_type in node_counts and node_counts[node_type] > 0:
                    self.log_test(
                        f"Node Type {node_type} Creation", 
                        True, 
                        f"Count: {node_counts[node_type]}"
                    )
                else:
                    self.log_test(
                        f"Node Type {node_type} Creation", 
                        False, 
                        "No nodes of this type found"
                    )
                    all_success = False
        else:
            self.log_test("Graph Statistics", False, f"Error: {result.get('error', 'Unknown')}")
            all_success = False
        
        # Test node retrieval
        for node_type in ['T', 'S', 'N', 'R']:
            result = self.make_request('GET', f'/api/graph/nodes/{node_type}')
            
            if 'error' not in result:
                nodes = result.get('nodes', [])
                self.log_test(
                    f"Retrieve {node_type} Nodes", 
                    True, 
                    f"Retrieved {len(nodes)} nodes"
                )
            else:
                self.log_test(
                    f"Retrieve {node_type} Nodes", 
                    False, 
                    f"Error: {result.get('error', 'Unknown')}"
                )
                all_success = False
        
        return all_success
    
    def test_search_functionality(self) -> bool:
        """Test various search capabilities."""
        all_success = True
        
        # Test entity search
        test_entities = ["AI", "Stanford University", "Dr. Sarah Chen", "machine learning"]
        
        for entity in test_entities:
            result = self.make_request('POST', '/api/search/entities', {
                'entity_name': entity
            })
            
            if 'error' not in result:
                count = result.get('count', 0)
                self.log_test(
                    f"Entity Search: {entity}", 
                    True, 
                    f"Found {count} matches"
                )
            else:
                self.log_test(
                    f"Entity Search: {entity}", 
                    False, 
                    f"Error: {result.get('error', 'Unknown')}"
                )
                all_success = False
        
        # Test advanced search (if available)
        test_queries = [
            "artificial intelligence applications",
            "neural networks research",
            "Stanford MIT collaboration"
        ]
        
        for query in test_queries:
            result = self.make_request('POST', '/api/search/advanced', {
                'query': query,
                'top_k': 5
            })
            
            if 'error' not in result:
                total_results = result.get('total_results', 0)
                self.log_test(
                    f"Advanced Search: {query[:30]}...", 
                    True, 
                    f"Found {total_results} results"
                )
            else:
                # Advanced search might not be available yet
                self.log_test(
                    f"Advanced Search: {query[:30]}...", 
                    False, 
                    f"Error: {result.get('error', 'Feature not available')}"
                )
                # Don't mark as failure if it's just not implemented
        
        return all_success
    
    def test_graph_analysis(self) -> bool:
        """Test graph analysis features."""
        all_success = True
        
        # Test communities
        result = self.make_request('GET', '/api/graph/communities')
        
        if 'error' not in result:
            communities = result.get('communities', [])
            self.log_test(
                "Community Detection", 
                True, 
                f"Found {len(communities)} communities"
            )
        else:
            self.log_test(
                "Community Detection", 
                False, 
                f"Error: {result.get('error', 'Unknown')}"
            )
            all_success = False
        
        # Test important entities
        result = self.make_request('GET', '/api/graph/important-entities')
        
        if 'error' not in result:
            entities = result.get('entities', [])
            self.log_test(
                "Important Entity Analysis", 
                True, 
                f"Found {len(entities)} important entities"
            )
        else:
            self.log_test(
                "Important Entity Analysis", 
                False, 
                f"Error: {result.get('error', 'Unknown')}"
            )
            all_success = False
        
        return all_success
    
    def test_visualization(self) -> bool:
        """Test graph visualization generation."""
        all_success = True
        
        # Test visualization stats
        result = self.make_request('GET', '/api/visualization/stats')
        
        if 'error' not in result:
            stats = result.get('stats', {})
            nodes = stats.get('total_nodes', 0)
            self.log_test(
                "Visualization Stats", 
                True, 
                f"Graph ready with {nodes} nodes"
            )
        else:
            self.log_test(
                "Visualization Stats", 
                False, 
                f"Error: {result.get('error', 'Unknown')}"
            )
            all_success = False
        
        # Test visualization creation
        result = self.make_request('POST', '/api/visualization/create', {
            'max_nodes': 500,
            'output_filename': 'test_visualization.html'
        })
        
        if 'error' not in result:
            filename = result.get('output_filename', '')
            self.log_test(
                "Graph Visualization Creation", 
                True, 
                f"Created: {filename}"
            )
        else:
            self.log_test(
                "Graph Visualization Creation", 
                False, 
                f"Error: {result.get('error', 'Unknown')}"
            )
            all_success = False
        
        # Test community visualization
        result = self.make_request('POST', '/api/visualization/community', {
            'output_filename': 'test_community_viz.html'
        })
        
        if 'error' not in result:
            communities = result.get('communities', 0)
            self.log_test(
                "Community Visualization", 
                True, 
                f"Visualized {communities} communities"
            )
        else:
            self.log_test(
                "Community Visualization", 
                False, 
                f"Error: {result.get('error', 'Unknown')}"
            )
            all_success = False
        
        return all_success
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        print("🚀 Starting RapidRFP RAG End-to-End Testing...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test API health first
        if not self.test_api_health():
            print("\n❌ API is not available. Please start the API server first.")
            return {'success': False, 'error': 'API not available'}
        
        # Create test documents
        print("\n📄 Creating test documents...")
        file_paths = self.create_test_documents()
        
        try:
            # Run all test suites
            print("\n🔄 Testing Document Processing...")
            doc_success = self.test_document_processing(file_paths)
            
            print("\n📊 Testing Graph Operations...")
            graph_success = self.test_graph_operations()
            
            print("\n🔍 Testing Search Functionality...")
            search_success = self.test_search_functionality()
            
            print("\n🌐 Testing Graph Analysis...")
            analysis_success = self.test_graph_analysis()
            
            print("\n🎨 Testing Visualization...")
            viz_success = self.test_visualization()
            
        finally:
            # Cleanup
            self.cleanup()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Summarize results
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        overall_success = failed_tests == 0
        
        if overall_success:
            print("\n🎉 ALL TESTS PASSED! The RapidRFP RAG system is working correctly.")
        else:
            print(f"\n⚠️  {failed_tests} tests failed. Please check the implementation.")
            print("\nFailed Tests:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  ❌ {result['test']}: {result['details']}")
        
        return {
            'success': overall_success,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'test_time': total_time,
            'test_results': self.test_results
        }

def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(description='End-to-end testing for RapidRFP RAG system')
    parser.add_argument(
        '--api-url', 
        default=TEST_API_URL,
        help=f'API base URL (default: {TEST_API_URL})'
    )
    parser.add_argument(
        '--output', 
        help='Output file for test results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = EndToEndTester(args.api_url)
    results = tester.run_comprehensive_test()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Test results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)

if __name__ == '__main__':
    main()