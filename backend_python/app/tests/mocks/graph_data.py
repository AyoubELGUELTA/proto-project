"""Données structurées simulant l'état du graphe entre les opérations."""

MOCK_ENTITY_SUMMARIZATION_INPUT = [
    {
        "title": "MUHAMMAD",
        "type": "PROPHET",
        "description": [
            "The final Messenger of Allah who migrated to Madinah.",
            "He was the military leader during the Battle of Badr.",
            "The Prophet established the Constitution of Madinah."
        ],
        "source_id": ["chunk_1", "chunk_2", "chunk_3"]
    },
    {
        "title": "MADINAH",
        "type": "CITY",
        "description": [
            "Formerly known as Yathrib.", 
            "The city that welcomed the Muhajirun."
        ],
        "source_id": ["chunk_1", "chunk_4"]
    }
]

MOCK_RELATIONSHIP_SUMMARIZATION_INPUT = [
    {
        "source": "MUHAMMAD",
        "target": "MADINAH",
        "description": ["Migration to the city.", "Establishment of leadership in the city."],
        "weight": 10.0,
        "source_id": ["chunk_1", "chunk_5"]
    }
]

MOCK_ENTITY_DESCRIPTIONS = [
    "The final Messenger of Allah who migrated to Madinah.",
    "He was the military leader during the Battle of Badr.",
    "The Prophet established the Constitution of Madinah.",
    "He is known as Al-Amin (The Trustworthy)."
]