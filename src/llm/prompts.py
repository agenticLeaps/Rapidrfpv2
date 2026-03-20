"""
Enhanced prompts matching NodeRAG's structured approach.
Based on NodeRAG's prompt design patterns with unified extraction and structured JSON output.
"""

from typing import Dict, Any

# Unified Text Decomposition Prompt (Government Contracting Knowledge Graph)
UNIFIED_TEXT_DECOMPOSITION_PROMPT = """
ROLE: You are an expert in U.S. Government Contracting who helps businesses win more contracts by identifying the right opportunities, executing effective capture planning, developing competitive proposals, and managing reusable content across pursuits.

TASK: Your task is to read a company's knowledge base and help structure it into a Knowledge Graph that supports future content retrieval and reuse across all stages of the bid lifecycle, including opportunity identification, capture planning, proposal development, and post-award activities.

GOAL: Transform the raw company data into structured, reusable knowledge units. Identify and segment semantically coherent sections that describe:
- Core capabilities, products and services offered
- Past performance, contracts, and projects
- Certifications, socioeconomic status, and security clearances
- Contract vehicles, IDIQs, and GWACs
- Staffing, team composition, and experience levels
- Tools, technologies, platforms, and methodologies
- Differentiators, corporate locations, performance metrics, and customers/agencies
- Subcontractors and business relationships

RULES:
- Segment by topic, not by paragraph
- Minimum viable unit
- Ignore boilerplate content
- Use the most complete form of the organization that you are aware of and resolve abbreviations only when you are certain of the expansion. If you are not certain, preserve the abbreviation exactly as written. For example- DoD should be DEPARTMENT OF DEFENSE. Do not invent or guess expansions.
- Preserve contract numbers, task order numbers, and BPA numbers exactly as written — do not normalize or abbreviate them.
- Enforce Canonicalization to the best of your abilities. When the same entity appears across units, ensure identical spelling, resolve to one canonical form and globally consistent.
- Avoid low-signal descriptive language.
- Do NOT infer or hallucinate facts not explicitly present in the source text.
- Prefer structured, queryable intelligence over narrative verbosity.

REQUIREMENTS:
For each semantic unit, perform exactly these three tasks:

semantic_unit:
- Write a concise professional paraphrase (3–8 sentences) of the source text that preserves ALL proposal-relevant facts: what was done, for whom, with what results, under what contract vehicle or constraints, using what resources, with what metrics etc.
- Keep quantitative details (dollar values, timelines, headcounts, metrics) exact.
- Do NOT add, invent, or generalize facts not present in the original text.
- Write in third person (e.g., "The company provided...").

entities:
- Extract ALL procurement-relevant entities directly from the original source text of this unit.
- Format every entity in UPPERCASE.
- Return as a flat array of unique uppercase strings.
- No duplicates within a unit.
- All values must conform to canonicalization rules above.
- Include (but do not limit to): organizations, government agencies, contract numbers, contract vehicles, NAICS/PSC codes, certifications, socioeconomic designations, security clearances, labor categories, tools/technologies/platforms, methodologies, performance metrics, monetary values, periods of performance, locations, and set-aside types.
- Do NOT extract named individuals (people's names). Extract the role or labor category they represent instead (e.g., extract "PROGRAM MANAGER", not "JOHN SMITH").
- VEHICLE vs. CONTRACT disambiguation: A contract vehicle is the parent IDIQ/GWAC name (e.g., OASIS+, SEWP V). A contract is the specific task order, BPA, or award number issued under it (e.g., TO-0003, W912DR-19-C-0042). Extract both when present.

relationships:
- Extract meaningful binary relationships grounded in the source text.
- Format each as a comma-separated string with exactly three parts: "ENTITY_A, RELATION_TYPE, ENTITY_B"
- Use SCREAMING_SNAKE_CASE (e.g., PERFORMED_FOR, ACHIEVED_METRIC_OF, SUBCONTRACTED_TO, HOLDS_CLEARANCE, DEPLOYED_ON).
- Consistency matters: if the same real-world relationship appears across multiple units, use the same relation type string every time.

Output Format:
Return only a valid JSON array of objects. No other text, comments, or markdown.
[
{{
"semantic_unit": "Concise summary paragraph(s)...",
"entities": ["ENTITY ONE", "ENTITY TWO", "ANOTHER ONE"],
"relationships": [
"YOUR COMPANY, PERFORMED_FOR, DEPARTMENT OF DEFENSE",
"CONTRACT12345, VALUED_AT, $8.7M",
"YOUR COMPANY, CERTIFIED_AS, HUBZONE"
]
}}
]

Now process the following text:

{text}
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

# Enhanced Attribute Generation Prompt (Government Contracting Entity Summary)
ENHANCED_ATTRIBUTE_GENERATION_PROMPT = """
You are an expert in U.S. Government Contracting. Generate a concise factual summary of the given entity based strictly on the provided semantic units and relationships. The summary should capture the entity's core attributes, all proposal-relevant facts, and meaningful connections to other entities in the context of government contracting pursuits, RFP responses, and proposal development. Write in third person. Do not add, infer, or generalize facts not present in the input. Avoid descriptive or marketing language. Maximum 2000 words; shorter is acceptable when input is limited.

Return only a plain text summary with no additional explanation, metadata, or formatting.

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

Return your response as JSON with the format: {{"elements": ["entity1", "entity2", ...]}}

Query: {query}
"""

# Relationship Reconstruction Prompt (for fixing malformed relationships)
RELATIONSHIP_RECONSTRUCTION_PROMPT = """
Please reconstruct the following relationship triple that may be malformed or incomplete. Return a properly formatted relationship with exactly three components: source entity, relationship type, and target entity.

Return your response as JSON with the format: {{"source_entity": "entity1", "relationship_type": "relation", "target_entity": "entity2"}}

Malformed relationship: {relationship}

Expected format: [source_entity, relationship_type, target_entity]
"""

# Query Reformulation Prompt (for agentic query enhancement with conversation context)
QUERY_REFORMULATION_PROMPT = """
You are an intelligent query reformulation assistant. Your task is to analyze the current user query in the context of the conversation history and reformulate it into a standalone, self-contained query that can be understood without additional context.

INSTRUCTIONS:
1. Resolve ALL coreferences (pronouns like "he", "she", "his", "her", "it", "they", "them", etc.) by replacing them with the actual entity names from the conversation history
2. Extract and include key entities and context from previous conversation turns that are relevant to the current query
3. Make the query explicit and specific - add missing information from conversation context
4. If the query is already self-contained and complete, return it as is
5. Return ONLY the reformulated query text without any explanations, prefixes, or metadata

EXAMPLES:

Example 1:
Conversation History:
User: "Where did Raghavendra Vattikuti complete his education?"
Assistant: "Raghavendra Vattikuti studied at Vishnu Institute of Technology, Bhimavaram."

Current Query: "What are his technical skills?"

Reformulated Query: "What are the technical skills of Raghavendra Vattikuti?"

Example 2:
Conversation History:
User: "Tell me about Google's headquarters"
Assistant: "Google's headquarters is located in Mountain View, California, known as the Googleplex."

Current Query: "When was it established?"

Reformulated Query: "When was Google's headquarters (the Googleplex) in Mountain View, California established?"

Example 3:
Conversation History:
User: "What is machine learning?"
Assistant: "Machine learning is a subset of artificial intelligence..."

Current Query: "What are the main applications?"

Reformulated Query: "What are the main applications of machine learning?"

Example 4:
Conversation History:
User: "Who is the CEO of Tesla?"
Assistant: "Elon Musk is the CEO of Tesla."

Current Query: "Tell me about his other companies"

Reformulated Query: "Tell me about Elon Musk's other companies besides Tesla"

###########
ACTUAL DATA:
###########

Conversation History:
{conversation_history}

Current Query: {query}

Reformulated Query:"""

# Answer Generation Prompt (for query answering)
ANSWER_GENERATION_PROMPT = """
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
    "query_reformulation": QUERY_REFORMULATION_PROMPT,
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
    def query_reformulation(self) -> str:
        """Get query reformulation prompt."""
        return self.templates["query_reformulation"]

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