#!/usr/bin/env python3
"""
Quick graph verification script
"""
import asyncio
from src.storage.neo4j_storage import Neo4jStorage

async def quick_verify():
    storage = Neo4jStorage()
    inspect_data = storage.inspect_all_data()
    
    if inspect_data.get('success'):
        results = inspect_data['results']
        node_types = {}
        for result in results:
            node_type = result['node_type']
            count = result['count']
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += count
        
        print("🔍 QUICK GRAPH VERIFICATION:")
        print("-" * 30)
        for node_type, count in node_types.items():
            print(f"   {node_type}: {count}")
        
        total = sum(node_types.values())
        entity_ratio = node_types.get('ENTITY', 0) / max(total, 1)
        rel_ratio = node_types.get('RELATIONSHIP', 0) / max(total, 1)
        
        print(f"\n✅ Quality Check:")
        print(f"   Entity ratio: {entity_ratio:.1%}")
        print(f"   Relationship ratio: {rel_ratio:.1%}")
        print(f"   Status: {'✅ GOOD' if entity_ratio > 0.08 and rel_ratio > 0.05 else '⚠️ CHECK NEEDED'}")

if __name__ == "__main__":
    asyncio.run(quick_verify())