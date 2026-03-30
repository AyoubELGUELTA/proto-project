import re
import unicodedata
import difflib

def similarity(a: str, b: str) -> float:
    """Calcule similarité entre 2 strings (0.0 à 1.0)"""
    return SequenceMatcher(None, a, b).ratio()

def normalize_entity_name(name: str) -> str:
    """
    Normalise un nom d'entité pour matching robuste.
    Version ULTRA-STRICTE v2.
    """
    if not name:
        return ""
    
    # 1. Strip + lowercase
    normalized = name.strip().lower()
    
    # 2. Supprime accents
    normalized = ''.join(
        c for c in unicodedata.normalize('NFD', normalized)
        if unicodedata.category(c) != 'Mn'
    )
    
    # 3. Supprime apostrophes et quotes
    normalized = re.sub(r"[''`´']", "", normalized)
    
    # 4. Supprime parenthèses
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)
    
    # 5. Supprime tirets et underscores
    normalized = re.sub(r'[-_]', ' ', normalized)
    
    # 6. Normalise "ibn", "bint", "al", "as"
    normalized = re.sub(r'\bbin\b', 'ibn', normalized)
    normalized = re.sub(r'\bal\s', '', normalized)  # Supprime "al " complètement
    normalized = re.sub(r'\bas\s', '', normalized)  # Supprime "as " complètement
    
    # 7. Normalise les doubles consonnes (yy → y, ss → s)
    normalized = normalized.replace('yy', 'y')
    normalized = normalized.replace('ss', 's')
    
    # 8. ch → sh (variante arabe)
    normalized = re.sub(r'ch\b', 'sh', normalized)
    
    # 9. Supprime espaces multiples
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized
