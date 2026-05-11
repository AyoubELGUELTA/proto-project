"""Réponses textuelles brutes simulées provenant du LLM."""

MOCK_SIRA_EXTRACTION_CHUNK_1 = """
("entity"<|>MUHAMMAD<|>Prophet<|>The final Messenger of Allah who migrated to Madinah)
##
("entity"<|>MADINAH<|>City<|>The city of the Prophet, formerly known as Yathrib)
##
("relationship"<|>MUHAMMAD<|>MADINAH<|>The Prophet performed the Hijra to Madinah in 622 CE<|>10)
<|COMPLETE|>
"""

MOCK_SIRA_EXTRACTION_CHUNK_2 = """
("entity"<|>MUHAMMAD<|>Prophet<|>The military leader during the Battle of Badr)
##
("entity"<|>BATTLE OF BADR<|>Battle<|>The first major battle between Muslims and Quraysh)
##
("relationship"<|>MUHAMMAD<|>BATTLE OF BADR<|>The Prophet commanded the Muslim army at Badr<|>9)
##
("relationship"<|>MUHAMMAD<|>UNKNOWN_PERSON<|>This relation should be filtered out because UNKNOWN is not an entity<|>5)
<|COMPLETE|>
"""

MOCK_SUMMARIZER_LLM_RESPONSE = (
    "Muhammad (PBUH) is the final Messenger of Allah who migrated from Makkah to Madinah (formerly Yathrib) in 622 CE. "
    "He established the Constitution of Madinah and served as the military commander during the pivotal Battle of Badr."
)