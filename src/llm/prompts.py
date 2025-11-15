"""
Enhanced prompts matching NodeRAG's structured approach.
Based on NodeRAG's prompt design patterns with unified extraction and structured JSON output.
"""

from typing import Dict, Any

# Unified Text Decomposition Prompt (matching NodeRAG's approach)
UNIFIED_TEXT_DECOMPOSITION_PROMPT = """
Goal: Given a text, segment it into multiple semantic units, each containing detailed descriptions of specific events or activities. 

Perform the following tasks:
1. Provide a summary for each semantic unit while retaining all crucial details relevant to the original context.
2. Extract all entities directly from the original text of each semantic unit, not from the paraphrased summary. Format each entity name in UPPERCASE. You should extract all entities including times, locations, people, organizations and all kinds of entities.
3. From the entities extracted in Step 2, list all relationships within the semantic unit and the corresponding original context in the form of string separated by comma: "ENTITY_A, RELATION_TYPE, ENTITY_B". The RELATION_TYPE could be a descriptive sentence, while the entities involved in the relationship must come from the entity names extracted in Step 2. Please make sure the string contains three elements representing two entities and the relationship type.

Requirements:
1. Temporal Entities: Represent time entities based on the available details without filling in missing parts. Use specific formats based on what parts of the date or time are mentioned in the text.

Each semantic unit should be represented as a dictionary containing three keys: semantic_unit (a paraphrased summary of each semantic unit), entities (a list of entities extracted directly from the original text of each semantic unit, formatted in UPPERCASE), and relationships (a list of extracted relationship strings that contain three elements, where the relationship type is a descriptive sentence). All these dictionaries should be stored in a list to facilitate management and access.

Example:

Text: In September 2024, Dr. Emily Roberts traveled to Paris to attend the International Conference on Renewable Energy. During her visit, she explored partnerships with several European companies and presented her latest research on solar panel efficiency improvements. Meanwhile, on the other side of the world, her colleague, Dr. John Miller, was conducting fieldwork in the Amazon Rainforest. He documented several new species and observed the effects of deforestation on the local wildlife. Both scholars' work is essential in their respective fields and contributes significantly to environmental conservation efforts.

Output:
[
  {{
    "semantic_unit": "In September 2024, Dr. Emily Roberts attended the International Conference on Renewable Energy in Paris, where she presented her research on solar panel efficiency improvements and explored partnerships with European companies.",
    "entities": ["DR. EMILY ROBERTS", "2024-09", "PARIS", "INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY", "EUROPEAN COMPANIES", "SOLAR PANEL EFFICIENCY"],
    "relationships": [
      "DR. EMILY ROBERTS, attended, INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY",
      "DR. EMILY ROBERTS, explored partnerships with, EUROPEAN COMPANIES",
      "DR. EMILY ROBERTS, presented research on, SOLAR PANEL EFFICIENCY"
    ]
  }},
  {{
    "semantic_unit": "Dr. John Miller conducted fieldwork in the Amazon Rainforest, documenting several new species and observing the effects of deforestation on local wildlife.",
    "entities": ["DR. JOHN MILLER", "AMAZON RAINFOREST", "NEW SPECIES", "DEFORESTATION", "LOCAL WILDLIFE"],
    "relationships": [
      "DR. JOHN MILLER, conducted fieldwork in, AMAZON RAINFOREST",
      "DR. JOHN MILLER, documented, NEW SPECIES",
      "DR. JOHN MILLER, observed the effects of, DEFORESTATION on LOCAL WILDLIFE"
    ]
  }},
  {{
    "semantic_unit": "The work of both Dr. Emily Roberts and Dr. John Miller is crucial in their respective fields and contributes significantly to environmental conservation efforts.",
    "entities": ["DR. EMILY ROBERTS", "DR. JOHN MILLER", "ENVIRONMENTAL CONSERVATION"],
    "relationships": [
      "DR. EMILY ROBERTS, contributes to, ENVIRONMENTAL CONSERVATION",
      "DR. JOHN MILLER, contributes to, ENVIRONMENTAL CONSERVATION"
    ]
  }}
]

#########
Real_Data:
#########
Text: {text}
"""

# Chinese version of unified text decomposition
UNIFIED_TEXT_DECOMPOSITION_PROMPT_CHINESE = """
目标：给定一个文本，将该文本被划分为多个语义单元，每个单元包含对特定事件或活动的详细描述。 
执行以下任务：
1.为每个语义单元提供总结，同时保留与原始上下文相关的所有关键细节。
2.直接从每个语义单元的原始文本中提取所有实体，而不是从改写的总结中提取。
3.从第2步中提取的实体中列出语义单元内的所有关系,其中关系类型可以是描述性句子。使用格式"ENTITY_A,RELATION_TYPE,ENTITY_B"，请确保字符串中包含三个元素，分别表示两个实体和关系类型。

要求：
时间实体：根据文本中提到的日期或时间的具体部分来表示时间实体，不填补缺失部分。

每个语义单元应以一个字典表示,包含三个键:semantic_unit(每个语义单元的概括性总结)、entities(直接从每个语义单元的原始文本中提取的实体列表,实体名格式为大写)、relationships(描述性句子形式的提取关系字符串三元组列表）。所有这些字典应存储在一个列表中，以便管理和访问。

实际数据： 
文本:{text} 
"""

# Enhanced Attribute Generation Prompt (matching NodeRAG's style)
ENHANCED_ATTRIBUTE_GENERATION_PROMPT = """
Generate a concise summary of the given entity, capturing its essential attributes and important relevant relationships. The summary should read like a character sketch in a novel or a product description, providing an engaging yet precise overview. Ensure the output only includes the summary of the entity without any additional explanations or metadata. The length must not exceed 2000 words but can be shorter if the input material is limited. Focus on distilling the most important insights with a smooth narrative flow, highlighting the entity's core traits and meaningful connections.

Entity: {entity}
Related Semantic Units: {semantic_units}
Related Relationships: {relationships}
"""

# Enhanced Community Summary Prompt (matching NodeRAG's specificity)
ENHANCED_COMMUNITY_SUMMARY_PROMPT = """
You will receive a set of text data from the same cluster. Your task is to extract distinct categories of high-level information, such as concepts, themes, relevant theories, potential impacts, and key insights. Each piece of information should include a concise title and a corresponding description, reflecting the unique perspectives within the text cluster.

Please do not attempt to include all possible information; instead, select the elements that have the most significance and diversity in this cluster. Avoid redundant information—if there are highly similar elements, combine them into a single, comprehensive entry. Ensure that the high-level information reflects the varied dimensions within the text, providing a well-rounded overview.

Clustered text data:
{content}
"""

# Query Decomposition Prompt (for search functionality)
QUERY_DECOMPOSITION_PROMPT = """
Please break down the following query into a single list. Each item in the list should either be a main entity (such as a key noun or object). If you have high confidence about the user's intent or domain knowledge, you may also include closely related terms. If uncertain, please only extract entities and semantic chunks directly from the query. Please try to reduce the number of common nouns in the list. Ensure all elements are organized within one unified list.

Query: {query}
"""

# Relationship Reconstruction Prompt (for fixing malformed relationships)
RELATIONSHIP_RECONSTRUCTION_PROMPT = """
Please reconstruct the following relationship triple that may be malformed or incomplete. Return a properly formatted relationship with exactly three components: source entity, relationship type, and target entity.

Malformed relationship: {relationship}

Expected format: [source_entity, relationship_type, target_entity]
"""

# Answer Generation Prompt (for query answering)
ANSWER_GENERATION_PROMPT = """
Based on the retrieved information below, please provide a comprehensive and accurate answer to the user's question. Use the information provided to construct a well-structured response that directly addresses the query.

Retrieved Information:
{info}

User Question: {query}

Please provide a detailed answer based on the retrieved information:
"""

# JSON Format Templates
JSON_FORMAT_TEMPLATES = {
    "text_decomposition": {
        "type": "object",
        "properties": {
            "Output": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "semantic_unit": {"type": "string"},
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "relationships": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["semantic_unit", "entities", "relationships"]
                }
            }
        },
        "required": ["Output"]
    },
    
    "query_decomposition": {
        "type": "object",
        "properties": {
            "elements": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["elements"]
    },
    
    "relationship_reconstruction": {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "relationship": {"type": "string"},
            "target": {"type": "string"}
        },
        "required": ["source", "relationship", "target"]
    }
}

# Prompt Templates Dictionary
PROMPT_TEMPLATES = {
    "unified_text_decomposition": UNIFIED_TEXT_DECOMPOSITION_PROMPT,
    "unified_text_decomposition_chinese": UNIFIED_TEXT_DECOMPOSITION_PROMPT_CHINESE,
    "enhanced_attribute_generation": ENHANCED_ATTRIBUTE_GENERATION_PROMPT,
    "enhanced_community_summary": ENHANCED_COMMUNITY_SUMMARY_PROMPT,
    "query_decomposition": QUERY_DECOMPOSITION_PROMPT,
    "relationship_reconstruction": RELATIONSHIP_RECONSTRUCTION_PROMPT,
    "answer_generation": ANSWER_GENERATION_PROMPT
}

class PromptManager:
    """
    Centralized prompt management matching NodeRAG's approach.
    Provides access to all prompts and their JSON format specifications.
    """
    
    def __init__(self, language: str = "english"):
        self.language = language.lower()
        self.templates = PROMPT_TEMPLATES
        self.json_formats = JSON_FORMAT_TEMPLATES
    
    @property
    def text_decomposition(self) -> str:
        """Get text decomposition prompt based on language."""
        if self.language == "chinese":
            return self.templates["unified_text_decomposition_chinese"]
        return self.templates["unified_text_decomposition"]
    
    @property
    def text_decomposition_json(self) -> Dict[str, Any]:
        """Get JSON format for text decomposition."""
        return self.json_formats["text_decomposition"]
    
    @property
    def attribute_generation(self) -> str:
        """Get attribute generation prompt."""
        return self.templates["enhanced_attribute_generation"]
    
    @property
    def community_summary(self) -> str:
        """Get community summary prompt."""
        return self.templates["enhanced_community_summary"]
    
    @property
    def query_decomposition(self) -> str:
        """Get query decomposition prompt."""
        return self.templates["query_decomposition"]
    
    @property
    def query_decomposition_json(self) -> Dict[str, Any]:
        """Get JSON format for query decomposition."""
        return self.json_formats["query_decomposition"]
    
    @property
    def relationship_reconstruction(self) -> str:
        """Get relationship reconstruction prompt."""
        return self.templates["relationship_reconstruction"]
    
    @property
    def relationship_reconstruction_json(self) -> Dict[str, Any]:
        """Get JSON format for relationship reconstruction."""
        return self.json_formats["relationship_reconstruction"]
    
    @property
    def answer_generation(self) -> str:
        """Get answer generation prompt."""
        return self.templates["answer_generation"]
    
    def format_prompt(self, prompt_key: str, **kwargs) -> str:
        """Format a prompt template with provided arguments."""
        if prompt_key not in self.templates:
            raise ValueError(f"Unknown prompt key: {prompt_key}")
        
        return self.templates[prompt_key].format(**kwargs)
    
    def get_json_format(self, format_key: str) -> Dict[str, Any]:
        """Get JSON format specification."""
        if format_key not in self.json_formats:
            raise ValueError(f"Unknown JSON format key: {format_key}")
        
        return self.json_formats[format_key]