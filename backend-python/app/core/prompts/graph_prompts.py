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

- entity_name: Name of the entity. 
  *STRICT RULE*: Use English names or English phonetic transliteration only.
- entity_type: One of the following types: [{entity_types}]
- entity_description: 
  A strictly factual summary of what the text says about this entity.
  Start directly with the fact (e.g., "Uncle of the Prophet", "Site of the burial").
  If the text only names the entity without details, leave the description empty or use "Mentioned in the text".
  *GROUDING* No external or general knowledge should be put if it is not mentioned in the text.
  
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. Identify all pairs of (source_entity, target_entity) that are *clearly related*.
For each pair, extract:
- source_entity: name as identified in step 1
- target_entity: name as identified in step 1
- relationship_description: 
  A single, concise sentence (MAX 15 words). Focus on the verb/action linking them.
  Bad: "Muhammad is related to Maymuna because he chose to marry her after the pilgrimage at a place called Sarif."
  Good: "Muhammad married Maymuna after the pilgrimage at Sarif."

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
("entity"<|>Muhammad ﷺ<|>Prophet<|>Organized the defense of the community in Madinah after the Hijra)
##
("entity"<|>Madinah<|>City<|>Location where the community was defended after the Hijra)
##
("entity"<|>Battle of Badr<|>Battle<|>Military conflict that occurred in the second year of Hijra)
##
("entity"<|>Hamza ibn Abd al-Muttalib<|>Sahabi<|>Participant in the Battle of Badr noted for bravery)
##
("relationship"<|>Muhammad ﷺ<|>Madinah<|>Muhammad organized the community defense in Madinah<|>10)
##
("relationship"<|>Hamza ibn Abd al-Muttalib<|>Battle of Badr<|>Hamza was a combatant during the Battle of Badr<|>9)
##
("relationship"<|>Muhammad ﷺ<|>Battle of Badr<|>The Battle of Badr occurred under the Prophet's leadership of the community<|>8)
<|COMPLETE|>

Example 2:
Entity_types: MotherBeliever, Prophet, City, Location, Sahabi, SacredText

Text:

Maymuna bint al-Harith était la dernière épouse du Prophète. Le mariage a eu lieu à Sarif, une localité située près de La Mecque, après que les musulmans ont quitté la ville. Son oncle, Al-'Abbas, a agi comme son tuteur pour cette union bénie par Allah. Des références à la piété des épouses se trouvent dans le Coran.

######################
Output:
("entity"<|>Maymuna bint al-Harith<|>MotherBeliever<|>The last wife of the Prophet)
##
("entity"<|>Muhammad<|>Prophet<|>The Prophet and husband of Maymuna bint al-Harith)
##
("entity"<|>Sarif<|>Location<|>Place near Mecca where the marriage occurred)
##
("entity"<|>Mecca<|>City<|>City that the Muslims left before the marriage at Sarif)
##
("entity"<|>Al-'Abbas<|>Sahabi<|>Uncle of Maymuna who acted as her guardian for the marriage)
##
("entity"<|>Allah<|>God<|>Source of blessing for the union between Maymuna and the Prophet)
##
("entity"<|>Quran<|>SacredText<|>Text containing references to the piety of the Prophet's wives)
##
("relationship"<|>Muhammad<|>Maymuna bint al-Harith<|>Muhammad married Maymuna bint al-Harith as his last wife<|>10)
##
("relationship"<|>Al-'Abbas<|>Maymuna bint al-Harith<|>Al-'Abbas acted as guardian for Maymuna during the marriage union<|>9)
##
("relationship"<|>Maymuna bint al-Harith<|>Sarif<|>The marriage took place in the location of Sarif<|>9)
##
("relationship"<|>Maymuna bint al-Harith<|>Mecca<|>Maymuna married near Mecca after the Muslims departed the city<|>8)
##
("relationship"<|>Allah<|>Maymuna bint al-Harith<|>The union of Maymuna was blessed by Allah<|>7)
##
("relationship"<|>Quran<|>Maymuna bint al-Harith<|>The Quran refers to the piety of wives including Maymuna<|>6)
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

OUTPUT FORMAT:

Limit to {max_length} words.

Highly dense, professional prose.
"""
RELATIONSHIP_SUMMARIZE_SYSTEM_PROMPT = """
You are a graph-optimization engine. Synthesize multiple descriptions of a relationship into a single, dense, and factual statement.

STRICT RULES:

1. NO INTRODUCTIONS: Start directly with the facts. Do not say "The relationship is..." or "Based on the data...".

2. ATOMIC SYNTHESIS: Combine overlapping information. If multiple sources say the same thing, state it once.

3. ONLY PROVIDED DATA: Never use external historical or religious knowledge. If the data is vague, stay vague.

4. CHRONOLOGICAL ORDER: If dates or phases (Makkah/Madinah) are present, follow the timeline.

5. ACTION-ORIENTED: Focus on the functional link (e.g., "A supported B during X", "A is the brother of B", "A fought alongside B at Y").

OUTPUT FORMAT:

Max {max_length} words.

Single paragraph.

Highly dense, professional prose.
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
- Return ONLY the exact title of the entities as provided in the list.
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
- If a candidate matches perfectly, return the SLUG (AN_EXEMPLE).
- If NONE of the candidates match the context of the text (e.g., it is a different person sharing the same name), you MUST return "NEW_ENTITY".

You must return ONLY a valid JSON object in this format:
{
  "choice": "CANDIDATE_SLUG_OR_NEW_ENTITY"
}

Example:
User:
Extracted Entity: Umar (Type: Person)
Context: He was a young boy who lived in the house of the Prophet after his father died, and the Prophet taught him how to eat from the dish.

Encyclopedia Candidates:
- SLUG: UMAR_IBN_KHATTAB | Name: Umar ibn al-Khattab | Summary: The second Caliph of Islam. | Properties 
- SLUG: UMAR_IBN_ABI_SALAMA | Name: Umar ibn Abi Salama | Summary: The stepson of the Prophet who lived in his household. | Properties

Assistant:
{
  "choice": "UMAR_IBN_ABI_SALAMA"
}
""" # TODO TODO TODO TODO TODO TODO TODO ADD PROPERTIES EXEMPLES IN THE PROMPT ABOVE

ANCHORING_RESOLUTION_USER_PROMPT = """Extracted Entity: {entity_title} (Type: {entity_type})
Context: {entity_context}

Encyclopedia Candidates:
{candidates_text}
"""