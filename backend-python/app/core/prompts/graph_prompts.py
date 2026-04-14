# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

GRAPH_EXTRACTION_SYSTEM_PROMPT = """
-Goal-
- Identify all entities and relationships from a text document using a provided list of entity types and the global document context.

STRICT LANGUAGE RULE:
- All extracted ENTITY NAMES and RELATIONSHIP TYPES must be in ENGLISH.

- If the source text is in French, translate the entity name (e.g., 'Dieu' -> 'God', 'La Mecque' -> 'Mecca', 'Prière' -> 'Prayer', etc.).

- For Arabic names, use standard English phonetic transliteration. 

GROUNDING RULE:
- Extract ONLY information explicitly stated in the provided text. Do not use your internal knowledge to add family members, dates, or events not mentioned in the text.
- If the text says "He went to a city", do not name the city "Medina" unless the text does.

-Document Context-
The following metadata provides the global context of the document this text belongs to:
{document_metadata}

-Steps-
1. Identify all entities. For each identified entity, extract:
- entity_name: Name of the entity, capitalized. 
  *STRICT RULE*: Use English names or English phonetic transliteration only.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Concise description (MAX 3-4 sentences). Focus on titles (Kunya), lineage (Nasab), and key historical events mentioned ONLY in this text. 
  *GROUNDING*: Do not use external knowledge.

Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. Identify all pairs of (source_entity, target_entity) that are *clearly related*.
For each pair, extract:
- source_entity: name as identified in step 1
- target_entity: name as identified in step 1
- relationship_description: Explanation of the connection (MAX 3-4 sentences). Use English only.
  *GROUNDING*: Extract only relationships explicitly stated or directly implied by the text.
- relationship_strength: numeric score (1-10)

Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. Return output in English using **##** as the list delimiter.
4. When finished, output <|COMPLETE|>

######################
-Examples-
######################

Example 1:
Entity_types: Prophet, Sahabi, City, Battle
Text:
After the Hijra to Madinah, the Prophet Muhammad ﷺ organized the defense of the community. In the second year, the Battle of Badr took place. Hamza ibn Abd al-Muttalib showed great bravery during this conflict.
######################
Output:
("entity"<|>MUHAMMAD<|>Prophet<|>The Prophet of Islam who led the community in Madinah)
##
("entity"<|>MADINAH<|>City<|>The city where the Prophet migrated during the Hijra)
##
("entity"<|>BATTLE OF BADR<|>Battle<|>A major military conflict in the second year of Hijra)
##
("entity"<|>HAMZA IBN ABD AL-MUTTALIB<|>Sahabi<|>A brave companion and uncle of the Prophet who fought at Badr)
##
("relationship"<|>MUHAMMAD<|>MADINAH<|>The Prophet migrated to and led the community in Madinah<|>10)
##
("relationship"<|>HAMZA IBN ABD AL-MUTTALIB<|>BATTLE OF BADR<|>Hamza was a key combatant in the Battle of Badr<|>9)
##
("relationship"<|>MUHAMMAD<|>BATTLE OF BADR<|>The Prophet commanded the forces during the Battle of Badr<|>9)
<|COMPLETE|>

Example 2:
Entity_types: MotherBeliever, Prophet, City, Location, Sahabi, SacredText
Text:
Maymuna bint al-Harith était la dernière épouse du Prophète. Le mariage a eu lieu à Sarif, une localité située près de La Mecque, après que les musulmans ont quitté la ville. Son oncle, Al-'Abbas, a agi comme son tuteur pour cette union bénie par Allah. Des références à la piété des épouses se trouvent dans le Coran.
######################
Output:
("entity"<|>MAYMUNA BINT AL HARITH<|>MotherBeliever<|>The last wife of the Prophet Muhammad and a prominent figure in the early Muslim community)
##
("entity"<|>MUHAMMAD<|>Prophet<|>The Prophet of Islam and husband of Maymuna bint al-Harith)
##
("entity"<|>SARIF<|>Location<|>A place near Mecca where the marriage of Maymuna and the Prophet was consummated)
##
("entity"<|>MECCA<|>City<|>The holy city from which the Muslims departed before the marriage at Sarif)
##
("entity"<|>AL 'ABBAS<|>Sahabi<|>The uncle of the Prophet who acted as the guardian for Maymuna during her marriage)
##
("entity"<|>ALLAH<|>God<|>The One True God in Islam who is mentioned as the source of blessing for the union)
##
("entity"<|>QURAN<|>SacredText<|>The holy book of Islam containing references to the piety of the Prophet's wives)
##
("relationship"<|>MUHAMMAD<|>MAYMUNA BINT AL HARITH<|>Muhammad married Maymuna bint al-Harith as his final wife<|>10)
##
("relationship"<|>AL 'ABBAS<|>MAYMUNA BINT AL HARITH<|>Al-'Abbas acted as the legal guardian for Maymuna during her marriage contract<|>9)
##
("relationship"<|>MAYMUNA BINT AL HARITH<|>SARIF<|>The marriage of Maymuna took place and was consummated in the location of Sarif<|>9)
##
("relationship"<|>MAYMUNA BINT AL HARITH<|>MECCA<|>Maymuna's marriage occurred in proximity to Mecca after the Muslims left the city<|>8)
##
("relationship"<|>ALLAH<|>MAYMUNA BINT AL HARITH<|>The union of Maymuna is described as being blessed by Allah<|>7)
<|COMPLETE|>

######################
"""

GRAPH_EXTRACTION_USER_PROMPT = """
-Real Data-
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = "MANY entities and relationships were missed. Based on the document context, continue extracting using the same format:\n"
LOOP_PROMPT = "Are there more entities or relationships to extract? Answer Y or N.\n"



ENTITY_SUMMARIZE_SYSTEM_PROMPT = """
You are a precision-oriented assistant. Synthesize multiple entity descriptions into a single, factual, third-person summary.
STRICT RULES:
1. NO EXTERNAL KNOWLEDGE: Use ONLY the provided descriptions. Do not add birth dates, historical facts, or titles not present in the text.
2. CONTRADICTIONS: If descriptions conflict, mention both possibilities neutrally.
3. STYLE: Professional, objective, and concise.
Limit to {max_length} words.
"""

RELATIONSHIP_SUMMARIZE_SYSTEM_PROMPT = """
You are a precision-oriented assistant. Synthesize multiple descriptions of a relationship between two entities into one.
STRICT RULES:
1. NO EXTERNAL KNOWLEDGE: Only describe the connection as it appears in the provided data.
2. FOCUS: Explain the nature, context, and duration of the link between the two subjects.
3. STYLE: Professional and factual. Avoid flowery language.
Limit to {max_length} words.
"""

COMMON_SUMMARIZE_USER_PROMPT = """
#######
-Data-
Subject: {target_name}
Description List: {description_list}
#######
Output:
"""


ENTITY_RESOLUTION_SYSTEM_PROMPT = """
You are an expert historian specializing in the Sira (biography of Prophet Muhammad ﷺ).
Your task is to identify if a list of entities are duplicates.

### Instructions:
1. Analyze the names and especially the CONTEXT (Nasab/lineage, titles/Kunya, specific events).
2. **Lineage is key**: "Zayd ibn Harithah" and "Zayd ibn Thabit" are DIFFERENT people.
3. **The Most Complete Name Wins**: The "Target Canonical Name" must be the most formal and descriptive.
4. **Be conservative**: If not 100% sure, do NOT merge.

### Output Rules:
- Format: (MERGE <|> "Original Name" <|> "Target Canonical Name")
- If no duplicates, output: <|NO_MERGE|>
"""

ENTITY_RESOLUTION_USER_PROMPT = """
### Category: {entity_type}
### Candidates to Evaluate:
{candidates}

Output:
"""



ANCHORING_RESOLUTION_SYSTEM_PROMPT = """You are an expert Islamic historian and Sira scholar. 
Your task is to resolve ambiguous entities extracted from a text by matching them to the correct entry in an established Encyclopedia.

You will be provided with:
1. The extracted entity (Title, Type, and Context from the text).
2. A list of possible matching Candidates from the encyclopedia.

Decision Rules:
- Analyze the context to determine which candidate is the exact match.
- If a candidate matches perfectly, return its ID.
- If NONE of the candidates match the context of the text (e.g., it is a different person sharing the same name), you MUST return "NEW_ENTITY".

You must return ONLY a valid JSON object in this format:
{
  "choice": "CANDIDATE_ID_OR_NEW_ENTITY"
}

Example:
User:
Extracted Entity: Umar (Type: Person)
Context: He was a young boy who lived in the house of the Prophet after his father died, and the Prophet taught him how to eat from the dish.

Encyclopedia Candidates:
- ID: UMAR_IBN_KHATTAB | Name: Umar ibn al-Khattab | Summary: The second Caliph of Islam.
- ID: UMAR_IBN_ABI_SALAMA | Name: Umar ibn Abi Salama | Summary: The stepson of the Prophet who lived in his household.

Assistant:
{
  "choice": "UMAR_IBN_ABI_SALAMA"
}
"""

ANCHORING_RESOLUTION_USER_PROMPT = """Extracted Entity: {entity_title} (Type: {entity_type})
Context: {entity_context}

Encyclopedia Candidates:
{candidates_text}
"""