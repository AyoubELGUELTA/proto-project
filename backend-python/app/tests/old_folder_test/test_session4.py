import asyncio
import httpx
import time
from datetime import datetime

# Questions de test
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "Quelles Mères des Croyants ont déjà migré en Abyssinie ?",
        "expected_type": "general",
        "expected_entities": []
    },
    {
        "id": 2,
        "question": "Comment Khadijah a été un soutien envers le Prophète ?",
        "expected_type": "entity_overview",
        "expected_entities": ["Khadija"]
    },
    {
        "id": 3,
        "question": "Sawda et Aisha se parlaient entre elles malgré leur différence d'âge ?",
        "expected_type": "relationship",
        "expected_entities": ["Sawda", "Aisha"]
    },
    {
        "id": 4,
        "question": "Durant l'Hégire, quelle Mère des Croyants étaient présentes ?",
        "expected_type": "temporal",
        "expected_entities": []
    },
    {
        "id": 5,
        "question": "Quels sont les noms complets et description des deux Mères des croyants qui ont pour prénom Zaynab ?",
        "expected_type": "entity_overview",
        "expected_entities": ["Zaynab"]
    },
    {
        "id": 6,
        "question": "Quelle a été la première Mère des Croyants ?",
        "expected_type": "temporal",
        "expected_entities": []
    },
    {
        "id": 7,
        "question": "Quelle a été la dernière Mère des Croyants ?",
        "expected_type": "temporal",
        "expected_entities": []
    },
    {
        "id": 8,
        "question": "Pourquoi le Prophète a-t-il décidé de se marier avec autant d'épouses à la fois ?",
        "expected_type": "concept",
        "expected_entities": ["Prophète"]
    },
    {
        "id": 9,
        "question": "Safiyya et Juwayriya avaient-elles des points communs dans leur parcours ?",
        "expected_type": "comparison",
        "expected_entities": ["Safiyya", "Juwayriya"]
    },
    {
        "id": 10,
        "question": "Umm Salama a-t-elle joué un rôle dans les décisions du Prophète ?",
        "expected_type": "relationship",
        "expected_entities": ["Umm Salama", "Prophète"]
    }
]


async def test_single_question(client, question_data):
    """Teste une seule question et collecte les metrics."""
    
    q_id = question_data["id"]
    question = question_data["question"]
    
    print(f"\n{'='*80}")
    print(f"Question {q_id} : {question}")
    print('='*80)
    
    start_time = time.time()
    
    try:
        response = await client.get(
            "http://localhost:8000/query",
            params={"question": question, "limit": 15},
            timeout=120.0  # 2 min timeout
        )
        
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Extraction metrics
            result = {
                "id": q_id,
                "question": question,
                "latency": round(latency, 2),
                "query_type": data.get("query_type", "unknown"),
                "confidence": data.get("confidence", 0),
                "strategy_used": data.get("strategy_used", "unknown"),
                "entities_detected": data.get("entities_detected", []),
                "entities_resolved": data.get("entities_resolved", []),
                "chunks_count": data.get("chunks_count", 0),
                "answer": data.get("answer", "")[:500],  # Premiers 500 chars
                "expected_type": question_data["expected_type"],
                "type_match": data.get("query_type") == question_data["expected_type"]
            }
            
            # Affichage résultats
            print(f"📊 Type détecté : {result['query_type']} (attendu: {result['expected_type']})")
            print(f"   {'✅' if result['type_match'] else '❌'} Classification")
            print(f"📊 Confiance : {result['confidence']:.2f}")
            print(f"🎯 Stratégie : {result['strategy_used']}")
            print(f"🔍 Entités résolues : {len(result['entities_resolved'])}")
            print(f"📦 Chunks : {result['chunks_count']}")
            print(f"⏱️ Latency : {result['latency']}s")
            print(f"\n📝 Réponse (extrait) :\n{result['answer'][:300]}...")
            
            return result
        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None


async def run_all_tests():
    """Lance tous les tests et génère un rapport."""
    
    print("\n" + "="*80)
    print("🧪 SESSION 4 - TESTS AUTOMATIQUES")
    print("="*80)
    print(f"📅 Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Nombre de questions : {len(TEST_QUESTIONS)}")
    print("="*80)
    
    results = []
    
    async with httpx.AsyncClient() as client:
        for question_data in TEST_QUESTIONS:
            result = await test_single_question(client, question_data)
            if result:
                results.append(result)
            
            # Pause entre questions
            await asyncio.sleep(2)
    
    # ═══════════════════════════════════════════════════════════
    # GÉNÉRATION RAPPORT
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("📊 RAPPORT FINAL")
    print("="*80)
    
    if not results:
        print("❌ Aucun résultat collecté")
        return
    
    # Metrics globales
    total = len(results)
    classification_correct = sum(1 for r in results if r["type_match"])
    avg_latency = sum(r["latency"] for r in results) / total
    avg_chunks = sum(r["chunks_count"] for r in results) / total
    avg_confidence = sum(r["confidence"] for r in results) / total
    
    # Distribution strategies
    strategy_dist = {}
    for r in results:
        strategy = r["strategy_used"]
        strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1
    
    # Affichage
    print(f"\n📈 MÉTRIQUES GLOBALES")
    print(f"   Questions testées : {total}/10")
    print(f"   Classification correcte : {classification_correct}/{total} ({classification_correct/total*100:.1f}%)")
    print(f"   Confiance moyenne : {avg_confidence:.2f}")
    print(f"   Latency moyenne : {avg_latency:.2f}s")
    print(f"   Chunks moyens : {avg_chunks:.1f}")
    
    print(f"\n📊 DISTRIBUTION STRATÉGIES")
    for strategy, count in strategy_dist.items():
        print(f"   {strategy}: {count} ({count/total*100:.1f}%)")
    
    print(f"\n📋 DÉTAIL PAR QUESTION")
    print(f"{'ID':<4} {'Type':<8} {'✓':<3} {'Strategy':<20} {'Chunks':<7} {'Latency':<8}")
    print("-"*80)
    
    for r in results:
        match_icon = "✅" if r["type_match"] else "❌"
        print(f"{r['id']:<4} {r['query_type']:<8} {match_icon:<3} {r['strategy_used']:<20} {r['chunks_count']:<7} {r['latency']:.1f}s")
    
    # Sauvegarde rapport
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"tests/rapport_session4_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SESSION 4 - RAPPORT DE TESTS\n")
        f.write("="*80 + "\n\n")
        
        for r in results:
            f.write(f"\nQuestion {r['id']} : {r['question']}\n")
            f.write(f"Type détecté : {r['query_type']} (attendu: {r['expected_type']})\n")
            f.write(f"Match : {'✅' if r['type_match'] else '❌'}\n")
            f.write(f"Stratégie : {r['strategy_used']}\n")
            f.write(f"Chunks : {r['chunks_count']}\n")
            f.write(f"Latency : {r['latency']}s\n")
            f.write(f"Réponse :\n{r['answer']}\n")
            f.write("-"*80 + "\n")
    
    print(f"\n💾 Rapport sauvegardé : {report_file}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(run_all_tests())