# Graph Operations Endpoints

Endpoints for graph management, analysis, and data retrieval.

## Get Graph Statistics

### `GET /api/graph/stats`

Get comprehensive statistics about the current graph.

**Request:**
```bash
curl http://localhost:5001/api/graph/stats
```

**Response:**
```json
{
  "total_nodes": 157,
  "total_edges": 298,
  "node_type_counts": {
    "T": 12,
    "S": 45,
    "N": 67,
    "R": 23,
    "A": 8,
    "H": 2,
    "O": 2
  },
  "num_communities": 3,
  "is_connected": true
}
```

**Status Codes:**
- `200 OK` - Statistics retrieved successfully
- `500 Internal Server Error` - Server error

---

## Get Nodes by Type

### `GET /api/graph/nodes/<type>`

Retrieve all nodes of a specific type.

**Parameters:**

| Parameter | Type | Description | Valid Values |
|-----------|------|-------------|--------------|
| `type` | string | Node type code | `T`, `S`, `N`, `R`, `A`, `H`, `O` |

**Request:**
```bash
curl http://localhost:5001/api/graph/nodes/N
```

**Response:**
```json
{
  "node_type": "N",
  "count": 67,
  "nodes": [
    {
      "id": "N_chunk1_0",
      "type": "N",
      "content": "John Smith",
      "metadata": {
        "chunk_id": "chunk1",
        "entity_index": 0,
        "source_file": "/path/to/document.pdf",
        "mentions": ["chunk1", "chunk3"],
        "node_type": "entity"
      }
    },
    {
      "id": "N_chunk1_1", 
      "type": "N",
      "content": "New York",
      "metadata": {
        "chunk_id": "chunk1",
        "entity_index": 1,
        "source_file": "/path/to/document.pdf",
        "mentions": ["chunk1"],
        "node_type": "entity"
      }
    }
  ]
}
```

**Node Type Reference:**

| Type | Description | Content Example |
|------|-------------|-----------------|
| `T` | Text chunks | Original document segments |
| `S` | Semantic units | "John discovered the artifact in Egypt" |
| `N` | Named entities | "John Smith", "Cairo", "Microsoft" |
| `R` | Relationships | "John Smith works_for Microsoft" |
| `A` | Attributes | Detailed entity summaries |
| `H` | High-level summaries | Community topic summaries |
| `O` | Overview titles | "Project Management Topics" |

**Status Codes:**
- `200 OK` - Nodes retrieved successfully
- `400 Bad Request` - Invalid node type
- `500 Internal Server Error` - Server error

---

## Get Specific Node

### `GET /api/graph/node/<id>`

Get detailed information about a specific node and its connections.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | string | Unique node identifier |

**Request:**
```bash
curl http://localhost:5001/api/graph/node/N_chunk1_0
```

**Response:**
```json
{
  "node": {
    "id": "N_chunk1_0",
    "type": "N",
    "content": "John Smith",
    "metadata": {
      "chunk_id": "chunk1",
      "entity_index": 0,
      "source_file": "/path/to/document.pdf",
      "mentions": ["chunk1", "chunk3"],
      "node_type": "entity"
    }
  },
  "connected_nodes": [
    {
      "id": "T_chunk1",
      "type": "T",
      "content": "John Smith is a senior software engineer who works at Microsoft in Seattle..."
    },
    {
      "id": "R_chunk1_0",
      "type": "R", 
      "content": "John Smith works_for Microsoft"
    },
    {
      "id": "A_N_chunk1_0",
      "type": "A",
      "content": "John Smith is a senior software engineer with over 10 years of experience..."
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Node retrieved successfully
- `404 Not Found` - Node not found
- `500 Internal Server Error` - Server error

---

## Get Communities

### `GET /api/graph/communities`

Get information about detected communities in the graph.

**Request:**
```bash
curl http://localhost:5001/api/graph/communities
```

**Response:**
```json
{
  "total_communities": 3,
  "communities": [
    {
      "community_id": 0,
      "node_count": 45,
      "nodes": [
        {
          "id": "S_chunk1_0",
          "type": "S",
          "content": "John Smith started working on the new project in Seattle..."
        },
        {
          "id": "N_chunk1_0", 
          "type": "N",
          "content": "John Smith"
        }
      ]
    },
    {
      "community_id": 1,
      "node_count": 32,
      "nodes": [
        {
          "id": "S_chunk5_2",
          "type": "S", 
          "content": "The marketing team launched a new campaign targeting millennials..."
        }
      ]
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Communities retrieved successfully
- `500 Internal Server Error` - Server error

---

## Get Important Entities

### `GET /api/graph/important-entities`

Get the most important entities based on graph centrality metrics.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `percentage` | float | 0.2 | Percentage of entities to return (0.0-1.0) |

**Request:**
```bash
curl http://localhost:5001/api/graph/important-entities?percentage=0.1
```

**Response:**
```json
{
  "percentage": 0.1,
  "count": 7,
  "entities": [
    {
      "id": "N_chunk1_0",
      "content": "John Smith", 
      "metadata": {
        "chunk_id": "chunk1",
        "entity_index": 0,
        "source_file": "/path/to/document.pdf",
        "mentions": ["chunk1", "chunk3", "chunk7"],
        "node_type": "entity"
      }
    },
    {
      "id": "N_chunk2_1",
      "content": "Microsoft",
      "metadata": {
        "chunk_id": "chunk2", 
        "entity_index": 1,
        "source_file": "/path/to/document.pdf",
        "mentions": ["chunk2", "chunk4", "chunk6"],
        "node_type": "entity"
      }
    }
  ]
}
```

**Importance Calculation:**
- Uses combination of betweenness centrality and k-core decomposition
- Entities with more connections and bridge roles score higher
- Entities mentioned across multiple chunks get boosted importance

**Status Codes:**
- `200 OK` - Important entities retrieved successfully
- `500 Internal Server Error` - Server error

---

## Save Graph

### `POST /api/graph/save`

Save the current graph to disk.

**Request:**
```bash
curl -X POST http://localhost:5001/api/graph/save \
  -H "Content-Type: application/json" \
  -d '{"filepath": "data/processed/my_graph.gpickle"}'
```

**Request Body (Optional):**
```json
{
  "filepath": "data/processed/custom_graph.gpickle"
}
```

**Response:**
```json
{
  "success": true,
  "filepath": "data/processed/graph.gpickle",
  "message": "Graph saved successfully"
}
```

**Status Codes:**
- `200 OK` - Graph saved successfully
- `500 Internal Server Error` - Save failed

---

## Load Graph

### `POST /api/graph/load`

Load a previously saved graph from disk.

**Request:**
```bash
curl -X POST http://localhost:5001/api/graph/load \
  -H "Content-Type: application/json" \
  -d '{"filepath": "data/processed/my_graph.gpickle"}'
```

**Request Body (Optional):**
```json
{
  "filepath": "data/processed/custom_graph.gpickle"
}
```

**Response:**
```json
{
  "success": true,
  "filepath": "data/processed/graph.gpickle", 
  "message": "Graph loaded successfully",
  "stats": {
    "total_nodes": 157,
    "total_edges": 298,
    "node_type_counts": {
      "T": 12,
      "S": 45,
      "N": 67,
      "R": 23,
      "A": 8,
      "H": 2,
      "O": 2
    },
    "num_communities": 3,
    "is_connected": true
  }
}
```

**Status Codes:**
- `200 OK` - Graph loaded successfully
- `404 Not Found` - Graph file not found
- `500 Internal Server Error` - Load failed

**Example Workflow:**

```python
import requests

# Save current graph
requests.post('http://localhost:5001/api/graph/save', 
             json={'filepath': 'backup_graph.gpickle'})

# Process another document
requests.post('http://localhost:5001/api/index/document',
             json={'file_path': '/path/to/another_doc.pdf'})

# Load previous graph if needed
requests.post('http://localhost:5001/api/graph/load',
             json={'filepath': 'backup_graph.gpickle'})
```