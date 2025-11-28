#!/usr/bin/env python3
"""
Quick graph quality verification after ultra-optimizations
"""

import asyncio
from src.storage.neo4j_storage import Neo4jStorage

async def verify_graph_quality():
    """Check if the graph is properly constructed with ultra-optimized settings"""
    
    print("🔍 GRAPH QUALITY VERIFICATION")
    print("="*50)
    
    storage = Neo4jStorage()
    
    # Get latest file stats
    try:
        org_id = "test_org"
        stats = storage.get_file_stats(org_id)
        
        print(f"📊 Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"📊 Total graphs: {stats.get('total_graphs', 0)}")
        
        # Get detailed data for verification
        inspect_data = storage.inspect_all_data()
        
        if inspect_data.get('success'):
            results = inspect_data['results']
            
            print("\n🎯 NODE TYPE BREAKDOWN:")
            node_types = {}
            for result in results:
                node_type = result['node_type']
                count = result['count']
                if node_type not in node_types:
                    node_types[node_type] = 0
                node_types[node_type] += count
            
            for node_type, count in node_types.items():
                print(f"   {node_type}: {count} nodes")
            
            print("\n✅ QUALITY INDICATORS:")
            
            # Check for essential node types
            has_entities = 'ENTITY' in node_types
            has_relationships = 'RELATIONSHIP' in node_types  
            has_semantic = 'SEMANTIC' in node_types
            has_text = 'TEXT' in node_types
            
            print(f"   Entity nodes: {'✅' if has_entities else '❌'}")
            print(f"   Relationship nodes: {'✅' if has_relationships else '❌'}")
            print(f"   Semantic nodes: {'✅' if has_semantic else '❌'}")
            print(f"   Text nodes: {'✅' if has_text else '❌'}")
            
            # Check ratios
            total_nodes = sum(node_types.values())
            if total_nodes > 0:
                entity_ratio = node_types.get('ENTITY', 0) / total_nodes
                rel_ratio = node_types.get('RELATIONSHIP', 0) / total_nodes
                
                print(f"\n📈 EXTRACTION RATIOS:")
                print(f"   Entity ratio: {entity_ratio:.2%}")
                print(f"   Relationship ratio: {rel_ratio:.2%}")
                
                if entity_ratio > 0.1 and rel_ratio > 0.05:
                    print("   ✅ Healthy extraction ratios")
                else:
                    print("   ⚠️  Low extraction ratios - may be over-optimized")
            
            # Recent extraction quality
            print(f"\n🕐 RECENT EXTRACTIONS:")
            recent_files = [r for r in results if r['node_type'] == 'ENTITY'][-3:]
            for result in recent_files:
                print(f"   File {result['file_id'][:8]}: {result['count']} entities")
                
        else:
            print(f"❌ Failed to inspect data: {inspect_data.get('error')}")
            
    except Exception as e:
        print(f"❌ Error during verification: {e}")

if __name__ == "__main__":
    asyncio.run(verify_graph_quality())