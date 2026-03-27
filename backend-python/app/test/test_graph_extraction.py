"""
Test extraction Graph RAG + Post-processing sur 5 chunks artificiels
"""

import asyncio
import json
from pathlib import Path

from app.ingestion.graph_extraction import extract_graph_from_chunks
from app.ingestion.graph_post_processing import post_process_graph_extraction

# ============================================================
# CHUNKS DE TEST
# ============================================================

CHUNK_1_FAMILLE = """
Aïcha bint Abi Bakr, de la tribu Quraysh (Banu Taym), a épousé le Prophète Muhammad (ﷺ) 
à Médine en l'an 2 après l'Hégire. Elle était la fille d'Abu Bakr al-Siddiq (رضي الله عنه), 
premier calife. Sa sœur Asma bint Abi Bakr était mariée à Zubayr ibn al-Awwam.

Aisha a participé à la bataille d'Uhud en 3 AH, où elle apportait 
de l'eau aux combattants blessés aux côtés d'Umm Sulaym.
"""

CHUNK_2_PRONOMS = """
La fille du Prophète (ﷺ), Fatima, a épousé son cousin Ali ibn Abi Talib à Médine. 
Leur mariage a eu lieu après la bataille de Badr. Elle a donné naissance à Hassan et Hussain, 
qui sont les petits-fils du Messager d'Allah (ﷺ). 

Fatima az-Zahra était connue pour sa piété. Son père lui rendait souvent visite.
"""

CHUNK_3_EVENEMENTS = """
Lors de la migration vers l'Abyssinie en 615 CE, avant l'Hégire, un groupe de Muhajirun 
a trouvé refuge auprès du Négus. Parmi eux se trouvaient Uthman ibn Affan et son épouse 
Ruqayya bint Muhammad.

Plus tard, en l'an 1 AH (622 CE), la grande Hijra vers Yathrib (Médine) a marqué le début 
du calendrier islamique. Le Prophète (ﷺ) et Abu Bakr se sont cachés dans la grotte de Thawr 
pendant trois jours.
"""

CHUNK_4_TRIBUS = """
Khadija bint Khuwaylid appartenait au clan Banu Asad, une branche de la tribu Quraysh. 
Elle était une riche marchande mecquoise. Son premier mariage avec le Prophète Muhammad (ﷺ), 
de Banu Hashim (autre sous-clan de Quraysh), a renforcé les liens entre les clans.

Hamza ibn Abdul-Muttalib, oncle paternel du Prophète (ﷺ) et membre de Banu Hashim, 
était surnommé "le Lion d'Allah".
"""

CHUNK_5_BATAILLE = """
À la bataille de Badr en l'an 2 AH (Ramadan), le Prophète Muhammad (ﷺ) a dirigé 313 musulmans 
contre l'armée mecquoise de Quraysh. Ali ibn Abi Talib a combattu vaillamment. 
Hamza ibn Abdul-Muttalib, commandant de l'aile gauche, a été un héros clé de cette victoire.

Cette bataille s'est déroulée près des puits de Badr, entre La Mecque et Médine. 
Bilal ibn Rabah a donné l'appel à la prière après la victoire. Soixante-dix prisonniers 
de Quraysh ont été capturés.
"""

IDENTITY_CONTEXT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 FICHE IDENTITÉ DU DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 TITRE: Les Mères des Croyants - Test Extraction
📖 TYPE: Biographie collective
🎯 SUJET: Récit détaillé des épouses du Prophète Muhammad (ﷺ)

STRUCTURE DU DOCUMENT:
- 1. Khadija bint Khuwaylid
- 2. Aïcha bint Abi Bakr
- 3. Fatima bint Muhammad 

🔑 THÈMES CLÉS: Mariage prophétique, Transmission hadith, Batailles, Tribus

🕌 CONTEXTE: Période prophétique (610-632 CE), Mecque et Médine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ============================================================
# FONCTION TEST COMPLÈTE
# ============================================================

async def test_extraction_with_postprocessing():
    """Test extraction + post-processing sur 5 chunks"""
    
    # Préparer chunks
    test_chunks = [
        {"chunk_id": "test_001", "text": CHUNK_1_FAMILLE, "heading_full": "Relations familiales"},
        {"chunk_id": "test_002", "text": CHUNK_2_PRONOMS, "heading_full": "Fatima az-Zahra"},
        {"chunk_id": "test_003", "text": CHUNK_3_EVENEMENTS, "heading_full": "Migrations"},
        {"chunk_id": "test_004", "text": CHUNK_4_TRIBUS, "heading_full": "Khadija et tribus"},
        {"chunk_id": "test_005", "text": CHUNK_5_BATAILLE, "heading_full": "Bataille de Badr"},
    ]
    
    # PHASE 1: Extraction
    print("\n" + "="*70)
    print("PHASE 1: EXTRACTION ENTITIES + RELATIONS")
    print("="*70)
    
    results = await extract_graph_from_chunks(
        chunks=test_chunks,
        identity_context=IDENTITY_CONTEXT,
        domain="sira"
    )
    
    # Save raw results
    output_dir = Path("app/test/output")
    output_dir.mkdir(exist_ok=True)
    
    raw_file = output_dir / "graph_extraction_RAW.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Résultats bruts sauvegardés: {raw_file}")
    
    # Analyse brute
    print_analysis("AVANT POST-PROCESSING", results["entities"], results["relations"])
    
    # PHASE 2: Post-processing
    print("\n" + "="*70)
    print("PHASE 2: POST-PROCESSING (DEDUP + VALIDATION)")
    print("="*70)
    
    entities_clean, relations_clean, validation = await post_process_graph_extraction(
        entities=results["entities"],
        relations=results["relations"]
    )
    
    # Save cleaned results
    cleaned_results = {
        "entities": entities_clean,
        "relations": relations_clean,
        "validation": validation,
        "stats": {
            "before": {
                "entities": len(results["entities"]),
                "relations": len(results["relations"])
            },
            "after": {
                "entities": len(entities_clean),
                "relations": len(relations_clean)
            },
            "reduction": {
                "entities": len(results["entities"]) - len(entities_clean),
                "relations": len(results["relations"]) - len(relations_clean)
            }
        }
    }
    
    clean_file = output_dir / "graph_extraction_CLEAN.json"
    with open(clean_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Résultats nettoyés sauvegardés: {clean_file}")
    
    # Analyse finale
    print_analysis("APRÈS POST-PROCESSING", entities_clean, relations_clean)
    
    # Comparaison avant/après
    print("\n" + "="*70)
    print("📊 COMPARAISON AVANT/APRÈS POST-PROCESSING")
    print("="*70)
    print(f"Entities: {len(results['entities'])} → {len(entities_clean)} "
          f"(-{len(results['entities']) - len(entities_clean)} duplicates)")
    print(f"Relations: {len(results['relations'])} → {len(relations_clean)} "
          f"(-{len(results['relations']) - len(relations_clean)} orphelines)")
    
    if validation["valid"]:
        print("\n✅ Validation théologique: PASSED")
    else:
        print("\n❌ Validation théologique: FAILED")
        print(f"   Erreurs: {len(validation['errors'])}")
    
    print("\n" + "="*70 + "\n")
    
    return cleaned_results


def print_analysis(title: str, entities: list, relations: list):
    """Print analyse entities + relations"""
    
    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print(f"{'='*70}")
    
    # Entity types
    entity_types = {}
    for e in entities:
        t = e.get("type", "Unknown")
        entity_types[t] = entity_types.get(t, 0) + 1
    
    print(f"\n🧬 ENTITIES ({len(entities)} total):")
    for t, count in sorted(entity_types.items(), key=lambda x: -x[1]):
        print(f"  {t:20s}: {count}")
    
    # Relation types
    relation_types = {}
    for r in relations:
        t = r.get("relation_type", "Unknown")
        relation_types[t] = relation_types.get(t, 0) + 1
    
    print(f"\n🔗 RELATIONS ({len(relations)} total):")
    for t, count in sorted(relation_types.items(), key=lambda x: -x[1]):
        print(f"  {t:20s}: {count}")
    
    # Duplicates (exact name match)
    entity_names = [e.get("name", "").lower() for e in entities]
    seen = {}
    for name in entity_names:
        seen[name] = seen.get(name, 0) + 1
    
    duplicates = {k: v for k, v in seen.items() if v > 1}
    
    print("\n⚠️  DUPLICATIONS EXACTES:")
    if duplicates:
        for name, count in sorted(duplicates.items(), key=lambda x: -x[1])[:5]:
            print(f"  '{name}': {count} occurrences")
    else:
        print("  Aucune duplication exacte")
    
    # Variantes
    aicha_variants = [e["name"] for e in entities 
                      if "aicha" in e["name"].lower() or "aisha" in e["name"].lower()]
    if aicha_variants:
        print(f"\n🔤 Variantes Aïcha: {aicha_variants}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    asyncio.run(test_extraction_with_postprocessing())