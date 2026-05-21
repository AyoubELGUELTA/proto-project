from app.indexing.operations.communities.utils import resolve_context_references

MOCK_ID_MAPPING = {
    "sub_communities": {
        1: "community-uuid-leaf-cluster-888",
        2: "community-uuid-external-media-999"
    },
    "entities": {
        1: "uuid-verdant-oasis-111",
        2: "hash-harmony-assembly-222"
    },
    "relationships": {
        1: "rel-uuid-oasismarch-991",
        2: "rel-uuid-oasisassembly-992"
    }
}

# =====================================================================
# 🧪 SUITE DE TESTS
# =====================================================================

def test_resolve_nominal_case():
    """Vérifie la résolution classique Entities et Relationships."""
    raw_finding = "Verdant Oasis Plaza acts as the physical hub. [Data: Entities (1); Relationships (1, 2)]"
    expected = (
        "Verdant Oasis Plaza acts as the physical hub. "
        "[Data: Entities (uuid-verdant-oasis-111); Relationships (rel-uuid-oasismarch-991, rel-uuid-oasisassembly-992)]"
    )
    assert resolve_context_references(raw_finding, MOCK_ID_MAPPING) == expected


def test_resolve_hierarchical_sub_communities():
    """
    🎯 NOUVEAU : Vérifie qu'un rapport parent citant une sous-communauté 
    est correctement résolu avec son ID global de cluster.
    """
    raw_finding = (
        "La macro-communauté se structure autour d'un noyau logistique hérité des sous-rapports. "
        "[Data: SubCommunities (1, 2); Entities (2)]"
    )
    
    expected = (
        "La macro-communauté se structure autour d'un noyau logistique hérité des sous-rapports. "
        "[Data: SubCommunities (community-uuid-leaf-cluster-888, community-uuid-external-media-999); "
        "Entities (hash-harmony-assembly-222)]"
    )
    
    result = resolve_context_references(raw_finding, MOCK_ID_MAPPING)
    assert result == expected, f"Hierarchical mismatch: {result}"
    print("✅ Test Hiérarchique SubCommunities : Réussi")


def test_resolve_sub_communities_with_more_tag():
    """🎯 NOUVEAU : Vérifie la robustesse du tag +more sur les SubCommunities."""
    raw_finding = "Analyse saturée de sous-graphes. [Data: SubCommunities (1, +more)]"
    expected = "Analyse saturée de sous-graphes. [Data: SubCommunities (community-uuid-leaf-cluster-888, +more)]"
    
    assert resolve_context_references(raw_finding, MOCK_ID_MAPPING) == expected
    print("✅ Test Limite SubCommunities (+more) : Réussi")


def test_resolve_empty_and_robustness():
    """Vérifie la robustesse aux chaînes vides."""
    assert resolve_context_references("", MOCK_ID_MAPPING) == ""
    assert resolve_context_references(None, MOCK_ID_MAPPING) is None


if __name__ == "__main__":
    print("▶️ Lancement de la suite de tests étendue...\n")
    try:
        test_resolve_nominal_case()
        test_resolve_hierarchical_sub_communities()
        test_resolve_sub_communities_with_more_tag()
        test_resolve_empty_and_robustness()
        print("\n🚀 TOUT EST AU VERT. Le Resolver gère la hiérarchie de bout en bout !")
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST : {e}")