import pytest
import os
import json
import shutil
from fastapi import UploadFile
from io import BytesIO
from app.api.v1.ingest import ingest_single_file
from app.services.database.postgres_client import PostgresClient
from app.services.database.schema import CREATE_SCHEMA_QUERY 
from app.services.llm.factory import LLMFactory

@pytest.mark.asyncio
async def test_full_ingest_flow_integration():
    """
    Test d'intégration global : Ingestion réelle + Extraction de Graphe + Export JSON.
    """
    # 1. Préparation du fichier d'entrée
    # On utilise le fichier Maymuna que tu as fourni
    path_to_real_pdf = "app/tests/data/input/test_full_little_syra_chapter.pdf" 
    
    if not os.path.exists(path_to_real_pdf):
        pytest.fail(f"Le fichier {path_to_real_pdf} est introuvable.")

    with open(path_to_real_pdf, "rb") as f:
        file_content = f.read()
    
    filename = "our_mother_maymuna_test.pdf"
    fake_file = UploadFile(
        filename=filename,
        file=BytesIO(file_content),
        size=len(file_content)
    )

    db = PostgresClient()
    await db.connect()
    
    try:

        LLMFactory._tracker.usage.total_tokens = 0
        LLMFactory._tracker.usage.total_cost = 0.0

        # Initialisation DB
        await db.execute(CREATE_SCHEMA_QUERY)
        await db.execute("DELETE FROM documents WHERE filename = $1", filename)

        print(f"\n🚀 Lancement de l'ingestion réelle pour : {filename}")
        
        # --- EXÉCUTION DU PIPELINE COMPLET ---
        result = await ingest_single_file(fake_file)
        
        doc_id = result["doc_id"]
        assert doc_id is not None
        assert result["status"] == "success"


        # --- RÉCUPÉRATION DU RAPPORT DE CONSO VIA LA FACTORY ---
        report = LLMFactory._tracker.get_report()
        print(f"\n💰 RAPPORT DE CONSOMMATION RÉEL : {report}")

        # On ajoute ces infos au JSON final
        result["consumption"] = {
            "tokens": LLMFactory._tracker.usage.total_tokens,
            "cost": LLMFactory._tracker.usage.total_cost,
            "summary": report
        }
        
        # --- ASSERTIONS POSTGRES (Structure) ---
        doc_status = await db.fetchval("SELECT status FROM documents WHERE doc_id = $1", doc_id)
        assert doc_status == "COMPLETED"
        
        content_count = await db.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = $1 AND chunk_type = 'CONTENT'", doc_id
        )
        assert content_count > 0
        print(f"✅ Pipeline terminé. Chunks CONTENT : {content_count}")

        # --- ASSERTIONS GRAPHE (Logique) ---
        assert "graph" in result, "Le résultat devrait contenir les données du graphe"
        entities = result["graph"]["entities"]
        relations = result["graph"]["relationships"]

        # On vérifie qu'on a bien extrait des données
        assert len(entities) > 0, "Aucune entité extraite du document"
        assert len(relations) > 0, "Aucune relation extraite du document"

        # Vérification de la résolution d'entité (Optionnel mais recommandé)
        # On cherche si Maymuna est présente (sous n'importe quelle forme résolue)
        maymuna_found = any("maymuna" in e["title"].lower() for e in entities)
        assert maymuna_found, "L'entité principale Maymuna n'a pas été trouvée ou résolue."

        print(f"✅ Graphe extrait : {len(entities)} entités, {len(relations)} relations.")

        # --- EXPORT JSON POUR VISUALISATION MANUELLE ---
        output_dir = "app/tests/data/output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{doc_id}.json")
        
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=4, ensure_ascii=False)
        
        print(f"📂 Résultats complets écrits dans : {output_path}")
        print(f"💡 Vérifie ce fichier pour valider la qualité de la résolution d'entités !")

    finally:
        await db.disconnect()