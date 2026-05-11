import asyncio
import json
from pathlib import Path
from app.services.llm_service import LLMService
from app.ingestion.graph_extraction_light import run_light_extraction 
from app.db.neo4j.refactor import run_graph_relation_optimization 
from app.ingestion.graph_post_processing import post_process_graph_extraction




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


async def test_extraction_light_vs_heavy():
    """
    Test miroir pour comparer l'efficacité réelle (Qualité vs Coût)
    """
    llm_service = LLMService()
    
    # Même Chunks que l'ancien test pour l'équité
    test_chunks = [
        {"chunk_id": "test_001", "text": CHUNK_1_FAMILLE},
        {"chunk_id": "test_002", "text": CHUNK_2_PRONOMS},
        {"chunk_id": "test_003", "text": CHUNK_3_EVENEMENTS},
        {"chunk_id": "test_004", "text": CHUNK_4_TRIBUS},
        {"chunk_id": "test_005", "text": CHUNK_5_BATAILLE},
    ]

    print("\n" + "="*70)
    print("🚀 PHASE 1: EXTRACTION LIGHT (1-PASS + CACHING)")
    print("="*70)

    # 1. On lance ta nouvelle extraction (celle avec les prompts à 1024+ tokens)
    entities_raw, relations_raw = await run_light_extraction(test_chunks, IDENTITY_CONTEXT)
    
    # Analyse brute pour le log
    print_analysis("BRUT LIGHT (Avant Clean)", entities_raw, relations_raw)

    print("\n" + "="*70)
    print("🚀 PHASE 2: POST-PROCESSING (Même logique que l'ancien)")
    print("="*70)

    # 2. ON UTILISE LA MÊME FONCTION QUE D'ANTAN
    # Note: Importe-la bien depuis ton module de post-processing
    entities_clean, relations_clean, validation = await post_process_graph_extraction(
        entities=entities_raw,
        relations=relations_raw
    )

    # 3. SAUVEGARDE POUR COMPARAISON FINALE
    output_dir = Path("app/test/output")
    output_dir.mkdir(exist_ok=True)
    
    light_final_results = {
        "entities": entities_clean,
        "relations": relations_clean,
        "validation": validation,
        "tokens_report": llm_service.get_session_report() # Crucial pour le prix !
    }

    with open(output_dir / "light_extraction_CLEAN.json", "w", encoding="utf-8") as f:
        json.dump(light_final_results, f, ensure_ascii=False, indent=2)

    #4. Normalisation Neo4j 
    await run_graph_relation_optimization(llm_service)

    print("\n" + "="*70)
    print("📊 RÉSULTAT DU MATCH")
    print("="*70)
    print(f"Ancien CLEAN : graph_extraction_CLEAN.json")
    print(f"Nouveau CLEAN : light_extraction_CLEAN.json")
    print(llm_service.get_session_report())



def print_analysis(title: str, entities: list, relations: list):
    """Affiche une analyse rapide des résultats pour le terminal"""
    print(f"\n📊 {title}")
    print(f"🧬 Entités : {len(entities)}")
    print(f"🔗 Relations : {len(relations)}")
    
    # Types d'entités (top 5)
    types = {}
    for e in entities:
        t = e.get("type", "Inconnu")
        types[t] = types.get(t, 0) + 1
    print(f"   Types : {dict(sorted(types.items(), key=lambda x: -x[1])[:5])}")

if __name__ == "__main__":
    # C'est cette ligne qui lance réellement le moteur !
    asyncio.run(test_extraction_light_vs_heavy())