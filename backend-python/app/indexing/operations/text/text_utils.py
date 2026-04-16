import re
import logging
import unicodedata
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

def similarity(a: str, b: str) -> float:
    """
    Calculates the structural similarity ratio between two strings.
    
    Uses Gestalt Pattern Matching (SequenceMatcher) to return a value 
    between 0.0 (totally different) and 1.0 (identical).
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def normalize_entity_name(name: str) -> str:
    """
    Ultra-strict normalization for entity matching (V2).
    
    This function is the 'matching key' generator. It strips accents, 
    standardizes Arabic patronymics, and removes transcriptions 
    variances to ensure that 'Al-Hussein' and 'Husayn' can be identified 
    as potential duplicates.
    
    Transformation steps:
    1. Lowercase & accent removal.
    2. Punctuation & Parenthesis stripping.
    3. Arabic-specific standardization (bin -> ibn, al- removal).
    4. Phonetic smoothing (sh -> ch, double consonant collapse).
    """
    if not name:
        return ""
    
    # 1. Base Cleanup
    normalized = name.strip().lower()
    
    # 2. Accent Removal (Decomposition)
    normalized = ''.join(
        c for c in unicodedata.normalize('NFD', normalized)
        if unicodedata.category(c) != 'Mn'
    )
    
    # 3. Structural Cleaning: Remove apostrophes, quotes and parentheses
    normalized = re.sub(r"[''`´']", "", normalized)
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)
    
    # 4. Delimiter Standardisation: Hyphens and underscores to space
    normalized = re.sub(r'[-_]', ' ', normalized)
    
    # 5. Arabic Patronymic Normalization
    # 'bin' becomes 'ibn' for consistency
    normalized = re.sub(r'\bbin\b', 'ibn', normalized)
    # Remove 'al ' and 'as ' prefixes which are often optional in queries
    normalized = re.sub(r'\bal\s+', '', normalized)  
    normalized = re.sub(r'\bas\s+', '', normalized)
    
    # 6. Phonetic & Transcription Smoothing
    # Collapsing double consonants (yy -> y, ss -> s) to catch transcription errors
    normalized = normalized.replace('yy', 'y')
    normalized = normalized.replace('ss', 's')
    # Standardizing 'sh' to 'ch' (French/English variance)
    normalized = re.sub(r'sh\b', 'ch', normalized)
    
    # 7. Whitespace Harmonization
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def normalize_entity_title(name: str) -> str:
    """
    Lightweight normalization for display titles.
    
    Used to standardize the 'Title' field in the database without 
    the aggressive phonetic changes used in the matching key.
    Ensures titles are uppercase and clean of special delimiters.
    """
    if not name:
        return ""
    
    # Uppercase & Accent stripping
    name = name.upper()
    name = ''.join(c for c in unicodedata.normalize('NFD', name)
                  if unicodedata.category(c) != 'Mn')
    
    # Replace separators with spaces
    name = re.sub(r'[-_]', ' ', name)
    
    # Final cleanup
    name = ' '.join(name.split())
    return name