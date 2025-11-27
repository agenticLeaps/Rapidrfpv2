# NodeRAG Complete Prompts Documentation
## All Prompts Used in Graph Creation, Embeddings, and Retrieval (A to Z)

This document contains every prompt template used throughout the NodeRAG system's pipeline, from document processing to answer generation.

---

## 📋 **Table of Contents**

1. [Phase 1: Graph Decomposition Prompts](#phase-1-graph-decomposition-prompts)
2. [Phase 2: Graph Augmentation Prompts](#phase-2-graph-augmentation-prompts)
3. [Phase 3: Embedding & Community Prompts](#phase-3-embedding--community-prompts)
4. [Search & Retrieval Prompts](#search--retrieval-prompts)
5. [Answer Generation Prompts](#answer-generation-prompts)
6. [Utility & Helper Prompts](#utility--helper-prompts)
7. [Agentic Enhancement Prompts](#agentic-enhancement-prompts)

---

## 🔄 **Phase 1: Graph Decomposition Prompts**

### 1. **Unified Text Decomposition Prompt** (Core NodeRAG)
**Purpose**: Break down text into semantic units with entities and relationships
**File**: `src/llm/prompts.py` line 9-60

```
Goal: Given a text, segment it into multiple semantic units, each containing detailed descriptions of specific events or activities. 

Perform the following tasks:
1. Provide a summary for each semantic unit while retaining all crucial details relevant to the original context.
2. Extract all entities directly from the original text of each semantic unit, not from the paraphrased summary. Format each entity name in UPPERCASE. You should extract all entities including times, locations, people, organizations and all kinds of entities.
3. From the entities extracted in Step 2, list all relationships within the semantic unit and the corresponding original context in the form of string separated by comma: "ENTITY_A, RELATION_TYPE, ENTITY_B". The RELATION_TYPE could be a descriptive sentence, while the entities involved in the relationship must come from the entity names extracted in Step 2. Please make sure the string contains three elements representing two entities and the relationship type.

Requirements:
1. Temporal Entities: Represent time entities based on the available details without filling in missing parts. Use specific formats based on what parts of the date or time are mentioned in the text.

Each semantic unit should be represented as a dictionary containing three keys: semantic_unit (a paraphrased summary of each semantic unit), entities (a list of entities extracted directly from the original text of each semantic unit, formatted in UPPERCASE), and relationships (a list of extracted relationship strings that contain three elements, where the relationship type is a descriptive sentence). All these dictionaries should be stored in a list to facilitate management and access.

[Example with Coitonic-style output format...]

Text: {text}
```

### 2. **Chinese Text Decomposition Prompt**
**Purpose**: Chinese language version of unified decomposition
**File**: `src/llm/prompts.py` line 63-77

```
目标：给定一个文本，将该文本被划分为多个语义单元，每个单元包含对特定事件或活动的详细描述。 
[Chinese translation of decomposition logic...]
文本:{text}
```

### 3. **Legacy Semantic Units Extraction**
**Purpose**: Backward compatibility for older extraction method
**File**: `src/llm/llm_service.py` line 57-68

```
Extract independent semantic units from the following text. Each unit should be a complete, standalone concept or event that can be understood without additional context.

Rules:
- Each unit should be 1-2 sentences maximum
- Units should be independent and self-contained
- Focus on key events, facts, or ideas
- Maximum {max_units} units
- Return as a JSON list of strings

Text: {text}

Semantic Units:
```

### 4. **Entity Extraction Prompt**
**Purpose**: Extract named entities from text chunks
**File**: `src/llm/llm_service.py` line 83-98

```
Extract named entities from the following text. Focus on:
- People (names, titles, roles)
- Places (locations, buildings, geographical features)
- Organizations (companies, institutions, groups)
- Objects (specific items, products, concepts)
- Events (specific named events, meetings, projects)

Rules:
- Return only the entity names, not descriptions
- Use the most specific form (e.g., "Dr. John Smith" not just "John")
- Maximum {max_entities} entities
- Return as a JSON list of strings

Text: {text}

Entities:
```

### 5. **Relationship Extraction Prompt**
**Purpose**: Extract relationships between identified entities
**File**: `src/llm/llm_service.py` line 116-129

```
Extract relationships between the given entities from the text. 

Entities: {entities_str}

Rules:
- Only use entities from the provided list
- Relationship format: (Entity1, Relationship, Entity2)
- Use clear, simple relationship terms (e.g., "works for", "located in", "created by")
- Maximum {max_relationships} relationships
- Return as JSON list of [entity1, relationship, entity2] arrays

Text: {text}

Relationships:
```

---

## 🔧 **Phase 2: Graph Augmentation Prompts**

### 6. **Enhanced Attribute Generation Prompt**
**Purpose**: Generate detailed descriptions for important entities
**File**: `src/llm/prompts.py` line 80-86

```
Generate a concise summary of the given entity, capturing its essential attributes and important relevant relationships. The summary should read like a character sketch in a novel or a product description, providing an engaging yet precise overview. Ensure the output only includes the summary of the entity without any additional explanations or metadata. The length must not exceed 2000 words but can be shorter if the input material is limited. Focus on distilling the most important insights with a smooth narrative flow, highlighting the entity's core traits and meaningful connections.

Entity: {entity}
Related Semantic Units: {semantic_units}
Related Relationships: {relationships}
```

### 7. **Legacy Entity Attributes Prompt**
**Purpose**: Backward compatibility for entity attribute generation
**File**: `src/llm/llm_service.py` line 363-374

```
Generate comprehensive attributes for the entity "{entity}" based on the provided context. Include:

- Key characteristics and properties
- Role and importance in the context
- Relationships with other entities
- Notable actions or events involving this entity
- Any other relevant information

Context:
{context}

Generate a detailed but concise summary (2-3 paragraphs) about {entity}:
```

### 8. **Relationship Reconstruction Prompt**
**Purpose**: Fix malformed relationship triples
**File**: `src/llm/prompts.py` line 108-116

```
Please reconstruct the following relationship triple that may be malformed or incomplete. Return a properly formatted relationship with exactly three components: source entity, relationship type, and target entity.

Return your response as JSON with the format: {"source_entity": "entity1", "relationship_type": "relation", "target_entity": "entity2"}

Malformed relationship: {relationship}

Expected format: [source_entity, relationship_type, target_entity]
```

---

## 🌐 **Phase 3: Embedding & Community Prompts**

### 9. **Enhanced Community Summary Prompt**
**Purpose**: Generate high-level summaries for node communities
**File**: `src/llm/prompts.py` line 89-96

```
You will receive a set of text data from the same cluster. Your task is to extract distinct categories of high-level information, such as concepts, themes, relevant theories, potential impacts, and key insights. Each piece of information should include a concise title and a corresponding description, reflecting the unique perspectives within the text cluster.

Please do not attempt to include all possible information; instead, select the elements that have the most significance and diversity in this cluster. Avoid redundant information—if there are highly similar elements, combine them into a single, comprehensive entry. Ensure that the high-level information reflects the varied dimensions within the text, providing a well-rounded overview.

Clustered text data:
{content}
```

### 10. **Community Overview Generation Prompt**
**Purpose**: Create short titles for community clusters
**File**: `src/llm/llm_service.py` line 410-414

```
Create a short, keyword-based title (3-8 words) that captures the main theme of this summary:

Summary: {community_summary}

Title:
```

---

## 🔍 **Search & Retrieval Prompts**

### 11. **Query Decomposition Prompt**
**Purpose**: Break down search queries into main entities
**File**: `src/llm/prompts.py` line 99-105

```
Please break down the following query into a single list. Each item in the list should either be a main entity (such as a key noun or object). If you have high confidence about the user's intent or domain knowledge, you may also include closely related terms. If uncertain, please only extract entities and semantic chunks directly from the query. Please try to reduce the number of common nouns in the list. Ensure all elements are organized within one unified list.

Return your response as JSON with the format: {"elements": ["entity1", "entity2", ...]}

Query: {query}
```

---

## 💬 **Answer Generation Prompts**

### 12. **Standard Answer Generation Prompt**
**Purpose**: Generate responses using retrieved context
**File**: `src/llm/prompts.py` line 119-135

```
You are a helpful AI assistant that ONLY uses the provided retrieved information to answer questions. Do NOT use any external knowledge or information not explicitly provided below.

IMPORTANT INSTRUCTIONS:
1. ONLY use information from the "Retrieved Information" section below
2. If the retrieved information doesn't contain enough detail to answer the question, say "I don't have enough information in the provided context to answer this question completely."
3. Do NOT make up or infer information that is not explicitly stated in the retrieved information
4. Be accurate and cite specific details from the provided information
5. If asked about something not covered in the retrieved information, clearly state that it's not available in the provided context

Retrieved Information:
{info}

User Question: {query}

Based ONLY on the retrieved information above, please provide your answer:
```

### 13. **Agentic Knowledge Discovery Prompt**
**Purpose**: Cognitive exploration for general knowledge queries
**File**: `api_service.py` line 652-668

```
You are an intelligent knowledge assistant exploring the user's data repository. The user wants to understand what information is available in their knowledge base.

CONTEXT INFORMATION:
{retrieved_info}

USER QUERY: {query}

TASK: Provide a comprehensive, well-structured overview of the available knowledge. Focus on:

1. **Main Topics & Themes**: What are the primary subjects covered?
2. **Key Entities**: What organizations, companies, people, or products are mentioned?
3. **Important Concepts**: What key processes, programs, or relationships are described?
4. **Data Scope**: What types of information and knowledge areas are represented?

Structure your response to give the user a clear understanding of their knowledge base content. Be specific about what information is available and how it could be useful.

RESPONSE:
```

### 14. **Conversation Enhancement Prompt**
**Purpose**: Refine answers with conversation history context
**File**: `api_service.py` line 696-708, 803-815

```
Given the previous conversation context and the answer below, please refine the answer to be more contextual and helpful:

Previous conversation:
{conversation_history}

Current question: {query}

Generated answer: {final_response}

Please provide a refined response that takes into account the conversation history:
```

---

## 🛠️ **Utility & Helper Prompts**

### 15. **Knowledge Discovery Keywords Detection**
**Purpose**: Identify when user wants general knowledge exploration
**File**: `api_service.py` line 531-535

```python
knowledge_discovery_keywords = [
    "data you have", "data u have", "tell about", "what do you know", 
    "what information", "what data", "overview", "summary", "available data",
    "knowledge base", "tell me about", "what's in", "content you have"
]
```

---

## 🎯 **Agentic Enhancement Prompts**

### 16. **Multi-Query Exploration Prompts**
**Purpose**: Agentic system uses multiple queries for comprehensive knowledge discovery
**File**: `api_service.py` line 555-575

```python
# Overview query for H, O type nodes
"overview summary main topics key information"

# Entity query for N type nodes  
"companies organizations entities names"

# Semantic query for S type nodes (uses original user query)
{query}
```

---

## 📊 **JSON Format Specifications**

### Text Decomposition JSON Schema
```json
{
    "type": "object",
    "properties": {
        "Output": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "semantic_unit": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "relationships": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["semantic_unit", "entities", "relationships"]
            }
        }
    },
    "required": ["Output"]
}
```

### Query Decomposition JSON Schema
```json
{
    "type": "object",
    "properties": {
        "elements": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["elements"]
}
```

### Relationship Reconstruction JSON Schema
```json
{
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "relationship": {"type": "string"},
        "target": {"type": "string"}
    },
    "required": ["source", "relationship", "target"]
}
```

---

## 🔄 **NodeRAG Pipeline Flow**

1. **Document Processing → Unified Text Decomposition Prompt**
2. **Entity Enhancement → Enhanced Attribute Generation Prompt**  
3. **Community Detection → Enhanced Community Summary Prompt**
4. **Search Query → Query Decomposition Prompt**
5. **Answer Generation → Standard/Agentic Answer Prompts**
6. **Conversation → Enhancement Prompt**

---

## 🚀 **Production Features**

- **Multi-language Support**: English + Chinese prompts
- **Agentic Intelligence**: Cognitive query understanding  
- **Structured JSON**: All outputs follow strict schemas
- **NodeRAG Compatible**: Matches original NodeRAG research architecture
- **Production Ready**: Error handling, fallbacks, validation

---

*This documentation covers every prompt used in the complete NodeRAG system pipeline from document ingestion to intelligent answer generation.*