# API Usage Examples

Complete examples demonstrating common workflows and use cases.

## Quick Start Example

### 1. Basic Document Processing

```python
import requests
import json

base_url = "http://localhost:5001"

# Health check
health = requests.get(f"{base_url}/health")
print("API Status:", health.json())

# Index a document
file_path = "/path/to/your/document.pdf"
response = requests.post(f"{base_url}/api/index/document",
                        json={"file_path": file_path})

if response.status_code == 200:
    result = response.json()
    print(f"Processed {result['chunks_processed']} chunks")
    print(f"Created {result['graph_stats']['total_nodes']} nodes")
else:
    print("Error:", response.json())
```

### 2. File Upload Endpoint (Form Body)

**Note:** Currently the API expects file paths, but here's how a file upload endpoint would work:

```python
# If you add file upload support, it would use form data:
files = {'file': open('/path/to/document.pdf', 'rb')}
response = requests.post(f"{base_url}/api/upload/document", files=files)
```

For now, use file paths:
```python
response = requests.post(f"{base_url}/api/index/document",
                        json={"file_path": "/absolute/path/to/document.pdf"})
```

## Complete Workflow Examples

### Document Analysis Pipeline

```python
import requests
import time

class RapidRFPClient:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
    
    def health_check(self):
        """Check if API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        return response.status_code == 200
    
    def estimate_cost(self, file_path):
        """Estimate processing cost before indexing"""
        response = requests.post(f"{self.base_url}/api/estimate/cost",
                               json={"file_path": file_path})
        response.raise_for_status()
        return response.json()
    
    def index_document(self, file_path):
        """Index a document through complete pipeline"""
        response = requests.post(f"{self.base_url}/api/index/document",
                               json={"file_path": file_path})
        response.raise_for_status()
        return response.json()
    
    def search_entities(self, entity_name):
        """Search for entities by name"""
        response = requests.post(f"{self.base_url}/api/search/entities",
                               json={"entity_name": entity_name})
        response.raise_for_status()
        return response.json()
    
    def get_graph_stats(self):
        """Get current graph statistics"""
        response = requests.get(f"{self.base_url}/api/graph/stats")
        response.raise_for_status()
        return response.json()
    
    def get_important_entities(self, percentage=0.2):
        """Get most important entities"""
        response = requests.get(f"{self.base_url}/api/graph/important-entities",
                              params={"percentage": percentage})
        response.raise_for_status()
        return response.json()

# Usage example
client = RapidRFPClient()

# Check API health
if not client.health_check():
    print("API is not healthy!")
    exit(1)

# Process a document
file_path = "/path/to/business_proposal.pdf"

# 1. Estimate cost first
estimate = client.estimate_cost(file_path)
print(f"File size: {estimate['file_size_mb']:.1f} MB")
print(f"Estimated chunks: {estimate['estimated_chunks']}")
print(f"Estimated processing time: ~{estimate['estimated_llm_calls'] * 2} seconds")

# 2. Proceed with indexing if acceptable
if estimate['file_size_mb'] < 10:  # Only process files under 10MB
    print("Starting document indexing...")
    result = client.index_document(file_path)
    print(f"Indexing completed in {result['processing_time']:.1f} seconds")
    print(f"Graph statistics: {result['graph_stats']}")
else:
    print("File too large, skipping...")

# 3. Analyze the results
stats = client.get_graph_stats()
print(f"\nGraph contains {stats['total_nodes']} nodes and {stats['total_edges']} edges")
print(f"Node distribution: {stats['node_type_counts']}")

# 4. Find important entities
important = client.get_important_entities(0.1)  # Top 10%
print(f"\nTop {important['count']} important entities:")
for entity in important['entities'][:5]:
    mentions = len(entity['metadata']['mentions'])
    print(f"- {entity['content']} (mentioned {mentions} times)")
```

### Multi-Document Processing

```python
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_document_batch(document_paths, max_workers=3):
    """Process multiple documents concurrently"""
    client = RapidRFPClient()
    
    def process_single_doc(file_path):
        try:
            # Estimate first
            estimate = client.estimate_cost(file_path)
            if estimate['file_size_mb'] > 20:
                return {'file': file_path, 'skipped': True, 'reason': 'Too large'}
            
            # Process document
            result = client.index_document(file_path)
            return {'file': file_path, 'success': True, 'result': result}
        
        except Exception as e:
            return {'file': file_path, 'error': str(e)}
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_doc, path): path 
            for path in document_paths
        }
        
        # Collect results
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)
            
            # Print progress
            if result.get('success'):
                stats = result['result']['graph_stats']
                print(f"✓ {os.path.basename(result['file'])}: {stats['total_nodes']} nodes")
            elif result.get('skipped'):
                print(f"- {os.path.basename(result['file'])}: {result['reason']}")
            else:
                print(f"✗ {os.path.basename(result['file'])}: {result['error']}")
    
    return results

# Process all PDFs in a directory
pdf_files = glob.glob("/path/to/documents/*.pdf")
results = process_document_batch(pdf_files[:10])  # Process first 10 files

# Summary
successful = sum(1 for r in results if r.get('success'))
failed = sum(1 for r in results if r.get('error'))
skipped = sum(1 for r in results if r.get('skipped'))

print(f"\nProcessing complete: {successful} successful, {failed} failed, {skipped} skipped")
```

### Entity Analysis Workflow

```python
def analyze_document_entities(file_path):
    """Complete entity analysis workflow"""
    client = RapidRFPClient()
    
    # 1. Process document
    print("Processing document...")
    result = client.index_document(file_path)
    
    # 2. Get all entities
    response = requests.get(f"{client.base_url}/api/graph/nodes/N")
    all_entities = response.json()
    
    print(f"Found {all_entities['count']} entities")
    
    # 3. Get important entities
    important = client.get_important_entities(0.15)  # Top 15%
    
    # 4. Analyze each important entity
    entity_analysis = {}
    for entity in important['entities']:
        entity_name = entity['content']
        
        # Search for mentions
        mentions = client.search_entities(entity_name)
        
        # Get detailed node info
        node_response = requests.get(f"{client.base_url}/api/graph/node/{entity['id']}")
        node_details = node_response.json()
        
        entity_analysis[entity_name] = {
            'mentions_count': len(entity['metadata']['mentions']),
            'connected_nodes': len(node_details['connected_nodes']),
            'has_attributes': any(
                node['type'] == 'A' for node in node_details['connected_nodes']
            )
        }
    
    # 5. Generate report
    print("\n=== Entity Analysis Report ===")
    for name, analysis in entity_analysis.items():
        print(f"\n{name}:")
        print(f"  - Mentioned {analysis['mentions_count']} times")
        print(f"  - Connected to {analysis['connected_nodes']} other nodes")
        if analysis['has_attributes']:
            print(f"  - Has detailed attribute summary")
    
    return entity_analysis

# Run analysis
analysis = analyze_document_entities("/path/to/company_report.pdf")
```

### Graph Exploration

```python
def explore_graph_interactively():
    """Interactive graph exploration"""
    client = RapidRFPClient()
    
    while True:
        print("\n=== Graph Explorer ===")
        print("1. View graph statistics")
        print("2. Search entities") 
        print("3. View communities")
        print("4. Get important entities")
        print("5. Explore node connections")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            stats = client.get_graph_stats()
            print(f"\nGraph Statistics:")
            print(f"Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")
            print(f"Communities: {stats['num_communities']}")
            for node_type, count in stats['node_type_counts'].items():
                print(f"  {node_type}: {count}")
        
        elif choice == "2":
            entity_name = input("Enter entity name to search: ").strip()
            if entity_name:
                results = client.search_entities(entity_name)
                print(f"\nFound {results['count']} matches for '{entity_name}':")
                for entity in results['entities']:
                    mentions = len(entity['metadata']['mentions'])
                    print(f"  - {entity['content']} (ID: {entity['id']}, {mentions} mentions)")
        
        elif choice == "3":
            response = requests.get(f"{client.base_url}/api/graph/communities")
            communities = response.json()
            print(f"\nFound {communities['total_communities']} communities:")
            for comm in communities['communities']:
                print(f"  Community {comm['community_id']}: {comm['node_count']} nodes")
        
        elif choice == "4":
            percentage = float(input("Enter percentage (0.0-1.0, default 0.2): ") or "0.2")
            important = client.get_important_entities(percentage)
            print(f"\nTop {important['count']} important entities:")
            for entity in important['entities']:
                mentions = len(entity['metadata']['mentions'])
                print(f"  - {entity['content']} ({mentions} mentions)")
        
        elif choice == "5":
            node_id = input("Enter node ID to explore: ").strip()
            if node_id:
                try:
                    response = requests.get(f"{client.base_url}/api/graph/node/{node_id}")
                    node_info = response.json()
                    print(f"\nNode: {node_info['node']['content']}")
                    print(f"Type: {node_info['node']['type']}")
                    print(f"Connected to {len(node_info['connected_nodes'])} nodes:")
                    for conn in node_info['connected_nodes'][:5]:
                        print(f"  - {conn['type']}: {conn['content'][:50]}...")
                except requests.exceptions.HTTPError:
                    print("Node not found!")
        
        elif choice == "6":
            break
        
        else:
            print("Invalid choice!")

# Start interactive exploration
explore_graph_interactively()
```

### Error Handling and Monitoring

```python
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_document_processor(file_paths, retry_attempts=3):
    """Process documents with comprehensive error handling"""
    client = RapidRFPClient()
    
    results = {
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    for file_path in file_paths:
        logger.info(f"Processing {file_path}")
        
        for attempt in range(retry_attempts):
            try:
                # Check file exists and is readable
                if not os.path.exists(file_path):
                    results['failed'].append({
                        'file': file_path,
                        'error': 'File not found'
                    })
                    break
                
                # Estimate cost
                estimate = client.estimate_cost(file_path)
                
                # Skip if too large
                if estimate['file_size_mb'] > 50:
                    results['skipped'].append({
                        'file': file_path,
                        'reason': f"File too large ({estimate['file_size_mb']:.1f} MB)"
                    })
                    break
                
                # Process document
                start_time = datetime.now()
                result = client.index_document(file_path)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                results['successful'].append({
                    'file': file_path,
                    'processing_time': processing_time,
                    'nodes_created': result['graph_stats']['total_nodes'],
                    'chunks_processed': result['chunks_processed']
                })
                
                logger.info(f"Successfully processed {file_path} in {processing_time:.1f}s")
                break
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {file_path}: {e}")
                
                if attempt == retry_attempts - 1:
                    results['failed'].append({
                        'file': file_path,
                        'error': str(e)
                    })
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {e}")
                results['failed'].append({
                    'file': file_path,
                    'error': f"Unexpected error: {str(e)}"
                })
                break
    
    # Generate summary report
    print(f"\n=== Processing Summary ===")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Skipped: {len(results['skipped'])}")
    
    if results['failed']:
        print("\nFailed files:")
        for item in results['failed']:
            print(f"  - {item['file']}: {item['error']}")
    
    return results

# Usage
document_list = ["/path/to/doc1.pdf", "/path/to/doc2.docx", "/path/to/doc3.txt"]
summary = robust_document_processor(document_list)
```

## Testing and Validation

### API Testing Script

```python
import unittest
import requests

class TestRapidRFPAPI(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:5001"
        self.test_file = "/path/to/test_document.pdf"
    
    def test_health_check(self):
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'healthy')
    
    def test_config_endpoint(self):
        response = requests.get(f"{self.base_url}/api/config")
        self.assertEqual(response.status_code, 200)
        config = response.json()
        self.assertIn('chunk_size', config)
    
    def test_cost_estimation(self):
        response = requests.post(f"{self.base_url}/api/estimate/cost",
                               json={"file_path": self.test_file})
        self.assertEqual(response.status_code, 200)
        estimate = response.json()
        self.assertIn('estimated_tokens', estimate)
    
    def test_document_indexing(self):
        response = requests.post(f"{self.base_url}/api/index/document",
                               json={"file_path": self.test_file})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['success'])
        self.assertGreater(result['graph_stats']['total_nodes'], 0)

if __name__ == '__main__':
    unittest.main()
```

This comprehensive set of examples covers all major API usage patterns and provides robust error handling for production use.