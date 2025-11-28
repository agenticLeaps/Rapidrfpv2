#!/usr/bin/env python3
"""
Simple Graph Visualization without external dependencies
Uses only built-in Python libraries + Flask for visualization
"""

import asyncio
import json
import threading
import webbrowser
import time
from flask import Flask, render_template_string
from src.storage.neo4j_storage import Neo4jStorage

app = Flask(__name__)

# Simple HTML template using only built-in libraries
SIMPLE_GRAPH_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>NodeRAG Graph Verification</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; flex-wrap: wrap; }
        .stat-box { background: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center; min-width: 120px; margin: 5px; }
        .stat-number { font-size: 24px; font-weight: bold; color: #1976d2; }
        .stat-label { font-size: 12px; color: #666; }
        .quality-indicator { display: inline-block; padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: bold; margin: 10px; }
        .quality-excellent { background: #4caf50; color: white; }
        .quality-good { background: #ff9800; color: white; }
        .quality-poor { background: #f44336; color: white; }
        .section { background: #fafafa; padding: 15px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #2196f3; }
        .node-type { display: inline-block; background: #fff; border: 2px solid #ddd; padding: 8px 12px; margin: 5px; border-radius: 15px; }
        .node-entity { border-color: #4caf50; color: #4caf50; }
        .node-relationship { border-color: #e91e63; color: #e91e63; }
        .node-semantic { border-color: #2196f3; color: #2196f3; }
        .node-text { border-color: #ff9800; color: #ff9800; }
        .node-attribute { border-color: #9c27b0; color: #9c27b0; }
        .performance-bar { background: #f0f0f0; height: 20px; border-radius: 10px; overflow: hidden; margin: 5px 0; }
        .performance-fill { background: #4caf50; height: 100%; transition: width 0.3s ease; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f5f5f5; font-weight: bold; }
        .refresh-btn { background: #2196f3; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 10px; }
        .refresh-btn:hover { background: #1976d2; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🕸️ NodeRAG Graph Verification</h1>
            <p>Analysis of your knowledge graph structure and quality</p>
            <div class="quality-indicator {{ quality_class }}">{{ quality_status }}</div>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{{ total_nodes }}</div>
                <div class="stat-label">Total Nodes</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{{ entities }}</div>
                <div class="stat-label">Entities</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{{ relationships }}</div>
                <div class="stat-label">Relationships</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{{ semantic_units }}</div>
                <div class="stat-label">Semantic Units</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{{ processing_time }}s</div>
                <div class="stat-label">Processing Time</div>
            </div>
        </div>

        <div class="section">
            <h3>📊 Node Type Distribution</h3>
            {% for node_type, count, percentage in node_breakdown %}
            <div class="node-type node-{{ node_type.lower() }}">
                <strong>{{ node_type }}:</strong> {{ count }} ({{ percentage }}%)
            </div>
            {% endfor %}
        </div>
        
        <div class="section">
            <h3>🎯 Quality Metrics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                <tr>
                    <td>Entity Coverage</td>
                    <td>{{ entity_ratio }}%</td>
                    <td>{{ '✅ Good' if entity_ratio|float > 8.0 else '⚠️ Low' }}</td>
                </tr>
                <tr>
                    <td>Relationship Density</td>
                    <td>{{ rel_ratio }}%</td>
                    <td>{{ '✅ Good' if rel_ratio|float > 5.0 else '⚠️ Low' }}</td>
                </tr>
                <tr>
                    <td>Graph Connectivity</td>
                    <td>{{ connectivity_score }}</td>
                    <td>{{ connectivity_status }}</td>
                </tr>
                <tr>
                    <td>Node Diversity</td>
                    <td>{{ node_diversity }} types</td>
                    <td>{{ '✅ Rich' if node_diversity > 3 else '⚠️ Limited' }}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h3>⚡ Performance Analysis</h3>
            <div>
                <strong>Phase 1 (Decomposition):</strong> {{ phase1_time }}s
                <div class="performance-bar">
                    <div class="performance-fill" style="width: {{ phase1_percent }}%"></div>
                </div>
            </div>
            <div>
                <strong>Phase 2 (Augmentation):</strong> {{ phase2_time }}s  
                <div class="performance-bar">
                    <div class="performance-fill" style="width: {{ phase2_percent }}%"></div>
                </div>
            </div>
            <div>
                <strong>Phase 3 (Embeddings):</strong> {{ phase3_time }}s
                <div class="performance-bar">
                    <div class="performance-fill" style="width: {{ phase3_percent }}%"></div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>🔍 Recent Extractions</h3>
            <table>
                <tr><th>File ID</th><th>Node Type</th><th>Count</th><th>Created</th></tr>
                {% for extraction in recent_extractions %}
                <tr>
                    <td>{{ extraction.file_id[:12] }}...</td>
                    <td>{{ extraction.node_type }}</td>
                    <td>{{ extraction.count }}</td>
                    <td>{{ extraction.created }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <div style="text-align: center; margin: 20px;">
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Data</button>
            <button class="refresh-btn" onclick="window.open('/raw', '_blank')">📄 Raw JSON Data</button>
        </div>
        
        <div class="section">
            <h3>💡 Recommendations</h3>
            <div id="recommendations">{{ recommendations | safe }}</div>
        </div>
    </div>
</body>
</html>
'''

def get_graph_analysis_sync():
    """Get comprehensive graph analysis data (synchronous version)"""
    try:
        storage = Neo4jStorage()
        
        # Get basic stats (using sync methods)
        stats = storage.get_file_stats("test_org")
        inspect_data = storage.inspect_all_data()
        
        if not inspect_data.get('success'):
            return {"error": "Failed to retrieve graph data"}
        
        results = inspect_data['results']
        node_types = {}
        recent_extractions = []
        
        # Process results
        for result in results:
            node_type = result['node_type']
            count = result['count']
            
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += count
            
            # Collect recent extractions
            recent_extractions.append({
                'file_id': result['file_id'],
                'node_type': node_type,
                'count': count,
                'created': result.get('last_created', 'N/A')
            })
        
        # Sort recent extractions
        recent_extractions = sorted(recent_extractions, 
                                  key=lambda x: x.get('created', ''), reverse=True)[:10]
        
        # Calculate metrics
        total_nodes = sum(node_types.values())
        entities = node_types.get('ENTITY', 0)
        relationships = node_types.get('RELATIONSHIP', 0)
        semantic_units = node_types.get('SEMANTIC', 0)
        
        entity_ratio = (entities / max(total_nodes, 1)) * 100
        rel_ratio = (relationships / max(total_nodes, 1)) * 100
        
        # Quality assessment
        if entity_ratio > 15 and rel_ratio > 10:
            quality_status = "Excellent Graph Quality"
            quality_class = "quality-excellent"
        elif entity_ratio > 8 and rel_ratio > 5:
            quality_status = "Good Graph Quality"
            quality_class = "quality-good"
        else:
            quality_status = "Graph Needs Review"
            quality_class = "quality-poor"
        
        # Connectivity score
        connectivity_score = min(100, (relationships / max(entities, 1)) * 100)
        connectivity_status = "✅ Well Connected" if connectivity_score > 30 else "⚠️ Sparse"
        
        # Node breakdown with percentages
        node_breakdown = []
        for node_type, count in node_types.items():
            percentage = (count / max(total_nodes, 1)) * 100
            node_breakdown.append((node_type, count, f"{percentage:.1f}"))
        
        # Generate recommendations
        recommendations = []
        if entity_ratio < 8:
            recommendations.append("• Consider increasing MAX_ENTITIES_PER_CHUNK for richer entity extraction")
        if rel_ratio < 5:
            recommendations.append("• Consider increasing MAX_RELATIONSHIPS_PER_CHUNK for better connectivity")
        if len(node_types) < 4:
            recommendations.append("• Graph has limited node diversity - check extraction pipeline")
        if total_nodes < 100:
            recommendations.append("• Small graph size - consider processing more content")
        
        if not recommendations:
            recommendations.append("• Graph structure looks healthy!")
            recommendations.append("• Good balance of entities and relationships")
            recommendations.append("• Extraction pipeline is working well")
        
        return {
            'total_nodes': total_nodes,
            'entities': entities,
            'relationships': relationships,
            'semantic_units': semantic_units,
            'entity_ratio': f"{entity_ratio:.1f}",
            'rel_ratio': f"{rel_ratio:.1f}",
            'quality_status': quality_status,
            'quality_class': quality_class,
            'connectivity_score': f"{connectivity_score:.1f}",
            'connectivity_status': connectivity_status,
            'node_diversity': len(node_types),
            'node_breakdown': node_breakdown,
            'recent_extractions': recent_extractions,
            'processing_time': 'Latest',
            'phase1_time': 'N/A',
            'phase2_time': 'N/A', 
            'phase3_time': 'N/A',
            'phase1_percent': 50,
            'phase2_percent': 30,
            'phase3_percent': 20,
            'recommendations': '<br>'.join(recommendations),
            'node_types': node_types
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def graph_view():
    """Main graph verification page"""
    try:
        data = get_graph_analysis_sync()
        
        if 'error' in data:
            return f"<h1>Error loading graph data</h1><p>{data['error']}</p><p>Details: Check if database is accessible and contains data.</p>"
        
        return render_template_string(SIMPLE_GRAPH_TEMPLATE, **data)
        
    except Exception as e:
        return f"<h1>Error in graph viewer</h1><p>{str(e)}</p>"

@app.route('/raw')  
def raw_data():
    """Raw JSON data endpoint"""
    try:
        data = get_graph_analysis_sync()
        return f"<pre>{json.dumps(data, indent=2)}</pre>"
    except Exception as e:
        return f"<pre>Error: {str(e)}</pre>"

def run_simple_server():
    """Run the simple graph server"""
    app.run(host='0.0.0.0', port=5004, debug=False, threaded=True)

def start_simple_graph_server():
    """Start the simple graph visualization server"""
    server_thread = threading.Thread(target=run_simple_server, daemon=True)
    server_thread.start()
    
    print("🕸️ Simple graph verification server starting...")
    print("🌐 Access your graph at: http://localhost:5004")
    
    # Wait and open browser
    time.sleep(2)
    try:
        webbrowser.open('http://localhost:5004')
        print("🚀 Browser opened automatically")
    except:
        print("💡 Manual access: http://localhost:5004")
    
    return "http://localhost:5004"

if __name__ == "__main__":
    start_simple_graph_server()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Server stopped")