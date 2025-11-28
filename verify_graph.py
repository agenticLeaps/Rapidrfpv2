#!/usr/bin/env python3
"""
Simple command-line graph verification
No web server required - just run and see results
"""

from src.storage.neo4j_storage import Neo4jStorage

def verify_graph():
    """Simple graph verification with detailed output"""
    print("🔍 NodeRAG Graph Verification")
    print("=" * 50)
    
    try:
        storage = Neo4jStorage()
        
        # Get data
        stats = storage.get_file_stats("test_org")
        inspect_data = storage.inspect_all_data()
        
        if not inspect_data.get('success'):
            print(f"❌ Failed to get graph data: {inspect_data.get('error', 'Unknown error')}")
            return
        
        results = inspect_data['results']
        node_types = {}
        
        # Process results
        for result in results:
            node_type = result['node_type']
            count = result['count']
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += count
        
        # Calculate totals
        total_nodes = sum(node_types.values())
        entities = node_types.get('ENTITY', 0)
        relationships = node_types.get('RELATIONSHIP', 0)
        semantic_units = node_types.get('SEMANTIC', 0)
        text_nodes = node_types.get('TEXT', 0)
        
        print(f"📊 GRAPH STATISTICS:")
        print(f"   Total Nodes: {total_nodes}")
        print(f"   Entities: {entities}")
        print(f"   Relationships: {relationships}")
        print(f"   Semantic Units: {semantic_units}")
        print(f"   Text Nodes: {text_nodes}")
        print()
        
        print(f"📈 NODE TYPE BREAKDOWN:")
        for node_type, count in sorted(node_types.items()):
            percentage = (count / max(total_nodes, 1)) * 100
            print(f"   {node_type}: {count} ({percentage:.1f}%)")
        print()
        
        # Quality analysis
        entity_ratio = (entities / max(total_nodes, 1)) * 100
        rel_ratio = (relationships / max(total_nodes, 1)) * 100
        connectivity = (relationships / max(entities, 1)) * 100 if entities > 0 else 0
        
        print(f"🎯 QUALITY ANALYSIS:")
        print(f"   Entity Coverage: {entity_ratio:.1f}%")
        print(f"   Relationship Density: {rel_ratio:.1f}%")
        print(f"   Connectivity Score: {connectivity:.1f}%")
        print(f"   Node Diversity: {len(node_types)} types")
        print()
        
        # Overall assessment
        print(f"✅ ASSESSMENT:")
        if entity_ratio > 15 and rel_ratio > 10:
            print(f"   🌟 EXCELLENT - Rich graph with good connectivity")
        elif entity_ratio > 8 and rel_ratio > 5:
            print(f"   ✅ GOOD - Healthy graph structure")
        elif entity_ratio > 3 and rel_ratio > 2:
            print(f"   ⚠️  FAIR - Graph has basic structure")
        else:
            print(f"   ❌ NEEDS REVIEW - Low extraction quality")
        
        print()
        
        # Recommendations
        print(f"💡 RECOMMENDATIONS:")
        if entity_ratio < 8:
            print(f"   • Increase MAX_ENTITIES_PER_CHUNK (currently extracting {entity_ratio:.1f}%)")
        if rel_ratio < 5:
            print(f"   • Increase MAX_RELATIONSHIPS_PER_CHUNK (currently {rel_ratio:.1f}%)")
        if len(node_types) < 4:
            print(f"   • Check extraction pipeline - only {len(node_types)} node types found")
        if total_nodes < 100:
            print(f"   • Consider processing more content - only {total_nodes} total nodes")
        if connectivity < 30:
            print(f"   • Improve relationship extraction - entities poorly connected")
        
        if entity_ratio > 8 and rel_ratio > 5 and len(node_types) >= 4:
            print(f"   ✅ Graph structure looks healthy!")
            print(f"   ✅ Good extraction balance achieved")
        
        print()
        print(f"🔗 Access detailed web view: python simple_graph_viewer.py")
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    verify_graph()