# Search & Query Endpoints

Endpoints for searching and retrieving information from the knowledge graph.

## Search Entities

### `POST /api/search/entities`

Search for entities by name with case-insensitive matching.

**Request:**
```bash
curl -X POST http://localhost:5001/api/search/entities \
  -H "Content-Type: application/json" \
  -d '{"entity_name": "John Smith"}'
```

**Request Body:**
```json
{
  "entity_name": "John Smith"
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entity_name` | string | Yes | Name of the entity to search for |

**Response:**
```json
{
  "query": "John Smith",
  "count": 2,
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
      "id": "N_chunk5_2",
      "content": "john smith",
      "metadata": {
        "chunk_id": "chunk5",
        "entity_index": 2,
        "source_file": "/path/to/document.pdf", 
        "mentions": ["chunk5"],
        "node_type": "entity"
      }
    }
  ]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | The search query that was executed |
| `count` | integer | Number of matching entities found |
| `entities` | array | List of matching entity nodes |

**Entity Object Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique entity identifier |
| `content` | string | Entity name/content |
| `metadata` | object | Entity metadata including mentions and source |

**Metadata Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | ID of the chunk where entity was first found |
| `entity_index` | integer | Index of entity within that chunk |
| `source_file` | string | Path to the source document |
| `mentions` | array | List of chunk IDs where entity is mentioned |
| `node_type` | string | Always "entity" for entity nodes |

**Search Features:**

- **Case Insensitive**: "john smith" matches "John Smith"
- **Exact Matching**: Searches for exact string matches
- **Deduplication**: Merged entities show all mention locations
- **Cross-Document**: Finds entities across multiple indexed documents

**Status Codes:**
- `200 OK` - Search completed successfully (even if no results)
- `400 Bad Request` - Missing or invalid entity_name
- `500 Internal Server Error` - Search error

**Example Usage:**

```python
import requests

# Search for a person
response = requests.post('http://localhost:5001/api/search/entities',
                        json={'entity_name': 'John Smith'})
results = response.json()

print(f"Found {results['count']} entities matching 'John Smith'")

for entity in results['entities']:
    mentions = len(entity['metadata']['mentions'])
    print(f"- {entity['content']} (mentioned {mentions} times)")
    print(f"  First seen in: {entity['metadata']['source_file']}")
```

**Advanced Search Patterns:**

```bash
# Search for organizations
curl -X POST http://localhost:5001/api/search/entities \
  -H "Content-Type: application/json" \
  -d '{"entity_name": "Microsoft"}'

# Search for locations  
curl -X POST http://localhost:5001/api/search/entities \
  -H "Content-Type: application/json" \
  -d '{"entity_name": "New York"}'

# Search for projects/concepts
curl -X POST http://localhost:5001/api/search/entities \
  -H "Content-Type: application/json" \
  -d '{"entity_name": "Project Apollo"}'
```

**Search Tips:**

1. **Exact Names**: Use the exact entity name as it appears in documents
2. **Case Flexibility**: Capitalization doesn't matter
3. **Partial Matches**: Currently only supports exact string matching
4. **Acronyms**: Search for both full names and acronyms separately
5. **Variations**: Try different name variations if no results found

**Entity Discovery Workflow:**

```python
# 1. Get all entities to browse available names
all_entities = requests.get('http://localhost:5001/api/graph/nodes/N').json()

# 2. Find interesting entity names
entity_names = [node['content'] for node in all_entities['nodes']]
print("Available entities:", entity_names[:10])

# 3. Search for specific entities
for name in ['Microsoft', 'John Smith', 'Seattle']:
    response = requests.post('http://localhost:5001/api/search/entities',
                           json={'entity_name': name})
    count = response.json()['count']
    print(f"{name}: {count} matches")
```

**Related Operations:**

After finding entities, you can:

1. **Get Details**: Use `/api/graph/node/<id>` to see connections
2. **Find Relationships**: Look for relationship nodes involving the entity
3. **Check Attributes**: Find attribute nodes describing important entities
4. **Explore Context**: Get the text chunks where the entity is mentioned

**Future Enhancements:**

- Fuzzy matching for typos and variations
- Semantic search using embeddings
- Relationship-based queries ("Find all people who work for Microsoft")
- Temporal queries ("Find entities mentioned in recent documents")
- Faceted search with filters by document, date, type, etc.