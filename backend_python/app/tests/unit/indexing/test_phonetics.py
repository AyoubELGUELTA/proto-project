import pytest
from phonetics import dmetaphone

def test_dmetaphone_arabic_variations():
    """
    Vérifie si Double Metaphone groupe correctement des variations 
    courantes de noms arabes.
    """
    # Test 1: Variations de terminaison (a vs ah)
    assert dmetaphone("Hamza")[0] == dmetaphone("Hamzah")[0]
    
    # Test 2: Variations de voyelles internes (o vs u)
    # Souvent, le premier code (Primary) est le même
    assert dmetaphone("Muhammad")[0] == dmetaphone("Mohammed")[0]
    
    # Test 3: Cas plus complexes (Kh vs H)
    # Note : Metaphone n'est pas parfait pour l'arabe, on vérifie ses limites
    khaled = dmetaphone("Khaled")
    halid = dmetaphone("Halid")
    print(f"\nPhonetic Khaled: {khaled}")
    print(f"Phonetic Halid: {halid}")

    # Test 4: Doubles consonnes
    assert dmetaphone("Abdullah")[0] == dmetaphone("Abdulla")[0]

def test_core_resolver_logic_sample():
    """
    Simule ce que ton CoreResolver devrait faire.
    """
    name1 = "Aisha"
    name2 = "Ayesha"
    
    # On compare les codes phonétiques
    # dmetaphone renvoie un tuple (Primary, Secondary)
    code1 = dmetaphone(name1)
    code2 = dmetaphone(name2)
    
    is_match = any(c1 == c2 and c1 is not None for c1 in code1 for c2 in code2)
    
    assert is_match, f"Phonetic mismatch: {code1} vs {code2}"
    print(f"✅ Aisha and Ayesha matched phonetically: {code1}")

if __name__ == "__main__":
    # Si tu lances le script directement sans pytest
    try:
        test_dmetaphone_arabic_variations()
        test_core_resolver_logic_sample()
        print("\n🚀 TOUS LES TESTS PHONÉTIQUES SONT AU VERT !")
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST : {e}")