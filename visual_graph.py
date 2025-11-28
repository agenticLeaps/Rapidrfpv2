#!/usr/bin/env python3
"""
Visual Network Graph Visualization (like Streamlit version)
Creates interactive network visualization without external dependencies
"""

import json
import threading
import webbrowser
import time
import math
import random
from flask import Flask, render_template_string
from src.storage.neo4j_storage import Neo4jStorage

app = Flask(__name__)

# Interactive Network Graph Template
NETWORK_GRAPH_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>NodeRAG Network Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #1a1a1a; color: white; }
        .container { padding: 20px; }
        .header { text-align: center; margin-bottom: 20px; }
        .controls { text-align: center; margin: 20px 0; }
        .control-btn { background: #4CAF50; color: white; border: none; padding: 8px 16px; margin: 5px; border-radius: 4px; cursor: pointer; }
        .control-btn:hover { background: #45a049; }
        .control-btn.active { background: #2196F3; }
        #network { width: 100%; height: 600px; border: 2px solid #333; border-radius: 8px; background: #000; position: relative; overflow: hidden; }
        .node { position: absolute; border-radius: 50%; cursor: pointer; transition: all 0.3s; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .node:hover { transform: scale(1.2); z-index: 100; }
        .edge { position: absolute; background: #666; height: 1px; transform-origin: left center; opacity: 0.6; }
        .legend { position: fixed; top: 20px; right: 20px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-color { width: 15px; height: 15px; border-radius: 50%; margin-right: 8px; }
        .stats { position: fixed; top: 20px; left: 20px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; font-size: 12px; }
        .tooltip { position: absolute; background: rgba(0,0,0,0.9); color: white; padding: 8px; border-radius: 4px; font-size: 12px; pointer-events: none; z-index: 1000; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🕸️ NodeRAG Network Graph</h1>
            <p>Interactive knowledge graph visualization</p>
        </div>
        
        <div class="controls">
            <button class="control-btn active" onclick="showAll()">All Nodes</button>
            <button class="control-btn" onclick="filterNodes('N')">Entities (N)</button>
            <button class="control-btn" onclick="filterNodes('R')">Relationships (R)</button>
            <button class="control-btn" onclick="filterNodes('S')">Semantic (S)</button>
            <button class="control-btn" onclick="filterNodes('A')">Attributes (A)</button>
            <button class="control-btn" onclick="randomizeLayout()">Randomize</button>
            <button class="control-btn" onclick="circleLayout()">Circle</button>
            <button class="control-btn" onclick="forceLayout()">Force</button>
        </div>
        
        <div id="network">
            <!-- Nodes and edges will be dynamically added here -->
        </div>
        
        <div class="stats">
            <div><strong>📊 Graph Stats</strong></div>
            <div>Total Nodes: <span id="total-nodes">{{ total_nodes }}</span></div>
            <div>Visible: <span id="visible-nodes">{{ total_nodes }}</span></div>
            <div>Entities: <span id="entity-count">{{ entities }}</span></div>
            <div>Relationships: <span id="rel-count">{{ relationships }}</span></div>
            <div>Connections: <span id="edge-count">{{ total_edges }}</span></div>
        </div>
        
        <div class="legend">
            <div><strong>🎨 Node Types</strong></div>
            <div class="legend-item"><div class="legend-color" style="background: #4CAF50;"></div>Entities (N)</div>
            <div class="legend-item"><div class="legend-color" style="background: #F44336;"></div>Relationships (R)</div>
            <div class="legend-item"><div class="legend-color" style="background: #2196F3;"></div>Semantic (S)</div>
            <div class="legend-item"><div class="legend-color" style="background: #9C27B0;"></div>Attributes (A)</div>
            <div class="legend-item"><div class="legend-color" style="background: #FF9800;"></div>High-level (H)</div>
            <div class="legend-item"><div class="legend-color" style="background: #795548;"></div>Overview (O)</div>
        </div>
        
        <div class="tooltip" id="tooltip"></div>
    </div>

    <script>
        // Graph data
        var graphData = {{ graph_data | safe }};
        var currentFilter = null;
        var networkDiv = document.getElementById('network');
        var tooltip = document.getElementById('tooltip');
        
        // Color mapping for node types
        var colors = {
            'N': '#4CAF50',  // Entities - Green
            'R': '#F44336',  // Relationships - Red  
            'S': '#2196F3',  // Semantic - Blue
            'A': '#9C27B0',  // Attributes - Purple
            'H': '#FF9800',  // High-level - Orange
            'O': '#795548'   // Overview - Brown
        };
        
        // Size mapping
        var baseSizes = {
            'N': 12, 'R': 8, 'S': 10, 'A': 6, 'H': 14, 'O': 16
        };
        
        function createNode(node, x, y) {
            var nodeEl = document.createElement('div');
            nodeEl.className = 'node';
            nodeEl.id = 'node-' + node.id;
            nodeEl.dataset.type = node.type;
            
            var size = baseSizes[node.type] || 8;
            var color = colors[node.type] || '#666';
            
            nodeEl.style.left = x + 'px';
            nodeEl.style.top = y + 'px';
            nodeEl.style.width = size + 'px';
            nodeEl.style.height = size + 'px';
            nodeEl.style.backgroundColor = color;
            nodeEl.textContent = node.type;
            
            // Tooltip
            nodeEl.addEventListener('mouseenter', function(e) {
                tooltip.innerHTML = '<strong>' + node.type + '</strong><br>' + 
                                 'ID: ' + node.id + '<br>' +
                                 'Content: ' + (node.content || 'N/A').substring(0, 50) + '...';
                tooltip.style.display = 'block';
                tooltip.style.left = (e.pageX + 10) + 'px';
                tooltip.style.top = (e.pageY - 10) + 'px';
            });
            
            nodeEl.addEventListener('mouseleave', function() {
                tooltip.style.display = 'none';
            });
            
            nodeEl.addEventListener('click', function() {
                // Highlight connected nodes
                highlightConnections(node.id);
            });
            
            return nodeEl;
        }
        
        function createEdge(x1, y1, x2, y2) {
            var edge = document.createElement('div');
            edge.className = 'edge';
            
            var length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
            var angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
            
            edge.style.left = x1 + 'px';
            edge.style.top = y1 + 'px';
            edge.style.width = length + 'px';
            edge.style.transform = 'rotate(' + angle + 'deg)';
            
            return edge;
        }
        
        function renderGraph() {
            // Clear network
            networkDiv.innerHTML = '';
            
            var nodes = graphData.nodes;
            var edges = graphData.edges;
            
            // Filter nodes if needed
            if (currentFilter) {
                nodes = nodes.filter(n => n.type === currentFilter);
            }
            
            // Update stats
            document.getElementById('visible-nodes').textContent = nodes.length;
            
            // Calculate layout bounds
            var width = networkDiv.clientWidth - 40;
            var height = networkDiv.clientHeight - 40;
            
            // Add nodes
            nodes.forEach((node, index) => {
                var x = node.x || (index % 10) * (width / 10) + 20;
                var y = node.y || Math.floor(index / 10) * (height / Math.ceil(nodes.length / 10)) + 20;
                
                // Ensure within bounds
                x = Math.max(20, Math.min(width, x));
                y = Math.max(20, Math.min(height, y));
                
                var nodeEl = createNode(node, x, y);
                networkDiv.appendChild(nodeEl);
                
                // Store position
                node.x = x;
                node.y = y;
            });
            
            // Add edges (sample for performance)
            var maxEdges = Math.min(edges.length, 200);
            for (var i = 0; i < maxEdges; i++) {
                var edge = edges[i];
                var sourceNode = nodes.find(n => n.id === edge.source);
                var targetNode = nodes.find(n => n.id === edge.target);
                
                if (sourceNode && targetNode && sourceNode.x && sourceNode.y && targetNode.x && targetNode.y) {
                    var edgeEl = createEdge(sourceNode.x, sourceNode.y, targetNode.x, targetNode.y);
                    networkDiv.appendChild(edgeEl);
                }
            }
        }
        
        function showAll() {
            currentFilter = null;
            updateActiveButton(0);
            renderGraph();
        }
        
        function filterNodes(type) {
            currentFilter = type;
            updateActiveButton(event.target);
            renderGraph();
        }
        
        function updateActiveButton(activeBtn) {
            document.querySelectorAll('.control-btn').forEach(btn => btn.classList.remove('active'));
            if (typeof activeBtn === 'number') {
                document.querySelectorAll('.control-btn')[activeBtn].classList.add('active');
            } else {
                activeBtn.classList.add('active');
            }
        }
        
        function randomizeLayout() {
            var width = networkDiv.clientWidth - 40;
            var height = networkDiv.clientHeight - 40;
            
            graphData.nodes.forEach(node => {
                node.x = Math.random() * width + 20;
                node.y = Math.random() * height + 20;
            });
            
            renderGraph();
        }
        
        function circleLayout() {
            var width = networkDiv.clientWidth - 40;
            var height = networkDiv.clientHeight - 40;
            var centerX = width / 2 + 20;
            var centerY = height / 2 + 20;
            var radius = Math.min(width, height) / 3;
            
            var visibleNodes = currentFilter ? 
                graphData.nodes.filter(n => n.type === currentFilter) : 
                graphData.nodes;
            
            visibleNodes.forEach((node, index) => {
                var angle = (index / visibleNodes.length) * 2 * Math.PI;
                node.x = centerX + radius * Math.cos(angle);
                node.y = centerY + radius * Math.sin(angle);
            });
            
            renderGraph();
        }
        
        function forceLayout() {
            // Simple force-directed layout simulation
            var width = networkDiv.clientWidth - 40;
            var height = networkDiv.clientHeight - 40;
            
            // Initialize random positions if not set
            graphData.nodes.forEach(node => {
                if (!node.x || !node.y) {
                    node.x = Math.random() * width + 20;
                    node.y = Math.random() * height + 20;
                }
                node.vx = 0;
                node.vy = 0;
            });
            
            // Run simulation
            for (var iter = 0; iter < 50; iter++) {
                // Repulsive forces between nodes
                for (var i = 0; i < graphData.nodes.length; i++) {
                    for (var j = i + 1; j < graphData.nodes.length; j++) {
                        var node1 = graphData.nodes[i];
                        var node2 = graphData.nodes[j];
                        var dx = node2.x - node1.x;
                        var dy = node2.y - node1.y;
                        var distance = Math.sqrt(dx*dx + dy*dy) + 0.01;
                        var force = 100 / distance;
                        
                        node1.vx -= force * dx / distance;
                        node1.vy -= force * dy / distance;
                        node2.vx += force * dx / distance;
                        node2.vy += force * dy / distance;
                    }
                }
                
                // Update positions
                graphData.nodes.forEach(node => {
                    node.x += node.vx * 0.1;
                    node.y += node.vy * 0.1;
                    
                    // Bounds checking
                    node.x = Math.max(20, Math.min(width, node.x));
                    node.y = Math.max(20, Math.min(height, node.y));
                    
                    // Damping
                    node.vx *= 0.9;
                    node.vy *= 0.9;
                });
            }
            
            renderGraph();
        }
        
        function highlightConnections(nodeId) {
            // Reset all nodes
            document.querySelectorAll('.node').forEach(n => n.style.opacity = '0.3');
            document.querySelectorAll('.edge').forEach(e => e.style.opacity = '0.1');
            
            // Highlight selected node
            var selectedNode = document.getElementById('node-' + nodeId);
            if (selectedNode) {
                selectedNode.style.opacity = '1';
                selectedNode.style.transform = 'scale(1.5)';
            }
            
            // Find and highlight connected nodes
            var connectedIds = new Set();
            graphData.edges.forEach(edge => {
                if (edge.source === nodeId) {
                    connectedIds.add(edge.target);
                } else if (edge.target === nodeId) {
                    connectedIds.add(edge.source);
                }
            });
            
            connectedIds.forEach(id => {
                var connectedNode = document.getElementById('node-' + id);
                if (connectedNode) {
                    connectedNode.style.opacity = '0.7';
                }
            });
            
            // Reset after 3 seconds
            setTimeout(() => {
                document.querySelectorAll('.node').forEach(n => {
                    n.style.opacity = '1';
                    n.style.transform = 'scale(1)';
                });
                document.querySelectorAll('.edge').forEach(e => e.style.opacity = '0.6');
            }, 3000);
        }
        
        // Initial render
        renderGraph();
        
        // Handle window resize
        window.addEventListener('resize', () => {
            setTimeout(renderGraph, 100);
        });
    </script>
</body>
</html>
'''

def get_network_data():
    """Get graph data and convert to network format"""
    try:
        storage = Neo4jStorage()
        inspect_data = storage.inspect_all_data()
        
        if not inspect_data.get('success'):
            return {"error": "Failed to get graph data"}
        
        # Convert database results to network format
        results = inspect_data['results']
        nodes = []
        edges = []
        node_types = {}
        
        # Create nodes from database (limit for performance)
        node_counter = 0
        for result in results:
            node_type = result['node_type']
            count = result['count']
            
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += count
            
            # Create sample nodes for visualization (limit to 500 for performance)
            for i in range(min(count, 20)):  # Max 20 nodes per type
                if node_counter < 500:  # Total limit
                    nodes.append({
                        'id': f"{node_type}_{i}",
                        'type': node_type,
                        'content': f"{node_type} node from {result['file_id'][:8]}",
                        'x': None,  # Will be set by layout
                        'y': None
                    })
                    node_counter += 1
        
        # Create sample edges (connections between different node types)
        for i in range(min(len(nodes) - 1, 300)):  # Limit edges
            if i < len(nodes) - 1:
                source = nodes[i]
                target = nodes[i + 1]
                
                # Create logical connections (entities to relationships, etc.)
                if (source['type'] == 'N' and target['type'] == 'R') or \
                   (source['type'] == 'S' and target['type'] == 'N') or \
                   (source['type'] == 'R' and target['type'] == 'A'):
                    edges.append({
                        'source': source['id'],
                        'target': target['id']
                    })
        
        # Add some random connections for visual interest
        for _ in range(min(50, len(nodes) // 4)):
            source = random.choice(nodes)
            target = random.choice(nodes)
            if source['id'] != target['id']:
                edges.append({
                    'source': source['id'],
                    'target': target['id']
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'total_nodes': sum(node_types.values()),
            'total_edges': len(edges),
            'entities': node_types.get('N', 0),
            'relationships': node_types.get('R', 0),
            'semantic': node_types.get('S', 0),
            'node_types': node_types
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def network_view():
    """Interactive network visualization"""
    try:
        data = get_network_data()
        
        if 'error' in data:
            return f"<h1>Network Error</h1><p>{data['error']}</p>"
        
        return render_template_string(NETWORK_GRAPH_TEMPLATE,
                                    graph_data=json.dumps({'nodes': data['nodes'], 'edges': data['edges']}),
                                    total_nodes=data['total_nodes'],
                                    entities=data['entities'],
                                    relationships=data['relationships'],
                                    total_edges=data['total_edges'])
        
    except Exception as e:
        return f"<h1>Visualization Error</h1><p>{str(e)}</p>"

def run_network_server():
    """Run the network visualization server"""
    app.run(host='0.0.0.0', port=5005, debug=False, threaded=True)

def start_network_graph():
    """Start the visual network graph"""
    server_thread = threading.Thread(target=run_network_server, daemon=True)
    server_thread.start()
    
    print("🕸️ Visual Network Graph starting...")
    print("🌐 Access at: http://localhost:5005")
    
    time.sleep(2)
    try:
        webbrowser.open('http://localhost:5005')
        print("🚀 Visual graph opened in browser")
    except:
        print("💡 Manual access: http://localhost:5005")
    
    return "http://localhost:5005"

if __name__ == "__main__":
    start_network_graph()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Network graph stopped")