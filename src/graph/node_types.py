from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

class NodeType(Enum):
    TEXT = "T"              # Text chunks
    SEMANTIC = "S"          # Semantic units
    ENTITY = "N"            # Named entities
    RELATIONSHIP = "R"      # Relationships
    ATTRIBUTE = "A"         # Entity attributes
    HIGH_LEVEL = "H"        # High-level summaries
    OVERVIEW = "O"          # Overview titles

@dataclass
class Node:
    id: str
    type: NodeType
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'content': self.content,
            'metadata': self.metadata,
            'embeddings': self.embeddings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        return cls(
            id=data['id'],
            type=NodeType(data['type']),
            content=data['content'],
            metadata=data['metadata'],
            embeddings=data.get('embeddings')
        )

@dataclass
class Edge:
    source: str
    target: str
    relationship_type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'relationship_type': self.relationship_type,
            'weight': self.weight,
            'metadata': self.metadata
        }