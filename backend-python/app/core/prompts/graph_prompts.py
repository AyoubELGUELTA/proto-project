# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Identify all entities and relationships from a text document using a provided list of entity types and the global document context.

-Document Context-
The following metadata provides the global context of the document this text belongs to:
{document_metadata}

-Steps-
1. Identify all entities. For each identified entity, extract:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description including, if available (for Human), titles (Kunya), lineage (Nasab), and key historical events they are associated with in this text.
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. Identify all pairs of (source_entity, target_entity) that are *clearly related*.
For each pair, extract:
- source_entity: name as identified in step 1
- target_entity: name as identified in step 1
- relationship_description: explanation of why they are related
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

######################
-Real Data-
######################
Entity_types: {entity_types}
Context: {document_metadata}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = "MANY entities and relationships were missed. Based on the document context, continue extracting using the same format:\n"
LOOP_PROMPT = "Are there more entities or relationships to extract? Answer Y or N.\n"

SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or more entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.
Limit the final description length to {max_length} words.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


ENTITY_RESOLUTION_PROMPT = """
You are an expert historian specializing in the Sira (biography of Prophet Muhammad ﷺ).
Your task is to identify if the following entities of type "{entity_type}" are duplicates.

### Instructions:
1. Analyze the names and especially the CONTEXT (Nasab/lineage, titles/Kunya, specific events).
2. **Lineage is key**: "Zayd ibn Harithah" and "Zayd ibn Thabit" are DIFFERENT people. Do not merge based on the first name only.
3. **The Most Complete Name Wins**: When merging, the "Target Canonical Name" must always be the most complete, formal, and descriptive name available in the batch.
4. **Be conservative**: If you are not 100% sure, or if the lineage (ibn...) differs, do NOT merge.
5. Format your output as a list of tuples, one per line:
(MERGE <|> "Original Name" <|> "Target Canonical Name")

### Candidates to Evaluate:
{candidates}

### Output Rules:
- Output ONLY the tuples.
- If no duplicates are found, output: <|NO_MERGE|>
- Use exactly the names provided in the list above.

Output:
"""