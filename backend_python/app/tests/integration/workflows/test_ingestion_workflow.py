import pytest
import os
from app.indexing.workflows.create_text_units import workflow_create_text_units
import re

@pytest.mark.asyncio
async def test_full_pdf_ingestion_workflow():
    """
    Test le pipeline complet : du PDF brut aux TextUnits raffinées et splittées.
    """
    pdf_path = "app/tests/data/input/R_GBA_27_EL-GUELTA_Ayoub_MOUNASSERE_Anwar.pdf"
    
    if not os.path.exists(pdf_path):
        pytest.fail(f"Le PDF de test est introuvable à l'adresse : {pdf_path}")


    identity_mock = """
    STRUCTURE DU DOCUMENT :
    - 1. Introduction
    - 2. Analyse technique
    - 3. Conclusion générale
    """
 
    units = await workflow_create_text_units(pdf_path, identity_text=identity_mock)
    
    print(f"\n" + "="*60)
    print(f"📊 RAPPORT D'INGESTION : {len(units)} unités générées")
    print("="*60)

    all_raw_headings = set()
    total_images = 0
    total_tables = 0

    for i, u in enumerate(units):
        # 1. Collecte pour résumé
        if u.headings: all_raw_headings.update(u.headings)
        img_count = len(u.metadata.get("image_urls", []))
        total_images += img_count
        table_count = len(u.tables)
        total_tables += table_count

        # 2. Print détaillé par unité (Format compact)
        # On affiche : [ID] [Page] [Titre Raffiné] [Nb Images/Tables]
        refined_h = u.metadata.get("heading_refined", "N/A")
        raw_h_count = len(u.headings)
        
        print(f"[{i:02d}] Page {u.page_numbers} | 🏷️ {refined_h[:40]:<40} | "
              f"Raw Headings: {raw_h_count} | 📸 {img_count} | 📊 {table_count}")
        
        # Optionnel : décommenter pour voir le début du texte de chaque chunk
        # print(f"     📝 Text: {u.text[:80].replace('\n', ' ')}...")

    print("="*60)
    print(f"📈 RÉSUMÉ GLOBAL :")
    print(f" - Titres uniques détectés (Raw) : {len(all_raw_headings)}")
    print(f" - Total images capturées        : {total_images}")
    print(f" - Total tableaux détectés       : {total_tables}")
    print("="*60 + "\n")