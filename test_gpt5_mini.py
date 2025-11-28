from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

test_text = """1.     Executive Summary
 
 
 
Summary of your understanding of Beebe’s system objectives and how your proposed solution will meet these objectives.
 
Based on our understanding of the requirements, Beebe Healthcare has a strategic plan aimed at delivering exceptional healthcare to Sussex County communities. To achieve this vision, Beebe recognizes the importance of a robust and integrated information technology (IT) ecosystem, referred to as One Beebe, which will benefit patients, families, providers, and staff. The current IT ecosystem requires significant enhancements to align with Beebe's strategic plan, support its growth and development over the next 3 to 5 years and beyond, and adapt to the future of healthcare.
 
Beebe's executive leadership acknowledges that the organization's position as an independent regional healthcare system relies heavily on the capabilities of its IT ecosystem. While some components of the current system have been in place for an extended period, they may have shortcomings. To leverage the potential of new disruptive technologies that can enhance Beebe's capabilities, it is crucial for Beebe to proactively enhance its IT ecosystem.
 
Therefore, Beebe seeks to incorporate innovative technologies that can act as catalysts for transformative healthcare change, leading to a distinctive One Beebe experience for both patients and providers. Beebe expects these innovations to be easily accessible not only to their known patients and providers but also to the wider community, enabling broad engagement with One Beebe healthcare and services. 
 
"""

prompt = f"Extract named entities from the following text. Focus on: - People (names, titles, roles) - Places (locations, buildings, geographical features) - Organizations (companies, institutions, groups) - Objects (specific items, products, concepts) - Events (specific named events, meetings, projects) Rules: - Return only the entity names, not descriptions - Use the most specific form  - Maximum 8 entities - Return as a JSON list of strings Text: {test_text} Entities:"

response = client.responses.create(
    model="gpt-5-nano-2025-08-07",
    input=prompt
)

print("Output:", response.output_text)
