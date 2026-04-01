import pytest
from app.models.domain import TextUnit
from app.indexing.operations.text.metadata_refiner import MetadataRefiner
from app.indexing.operations.text.text_splitter import TextSplitter

def test_metadata_refiner_inheritance():
    # Simulation de chunks avec titres bizarres
    units = [
        TextUnit(id="1", text="Intro", headings=["Chapitre 1"]),
        TextUnit(id="2", text="Suite", headings=["Page 12"]), # Titre suspect (Page 12)
    ]
    refiner = MetadataRefiner()
    refined = refiner.refine_units(units)
    
    assert refined[0].metadata["heading_refined"] == "Chapitre 1"
    assert refined[1].metadata["heading_refined"] == "Chapitre 1" # Héritage réussi

def test_text_splitter_table_flags():
    # Simulation d'un gros tableau
    big_table = "| col1 | col2 |\n" + ("| data | data |\n" * 50)
    unit = TextUnit(id="3", text=big_table, headings=["Tableau"])
    
    splitter = TextSplitter(max_tokens=100, overlap=30) 
    splitted = splitter.split_units([unit])

    print(f"\n--- DEBUG TEST ---")
    print(f"Nombre de chunks: {len(splitted)}")
    print(f"Chunk 0 metadata: {splitted[0].metadata}")
    print(f"Chunk 0 text (10 premiers car.): '{splitted[0].text[:10]}'")
    print(f"------------------")
    assert len(splitted) > 1
    assert splitted[0].metadata["is_table_cut"] is True
    assert splitted[1].metadata["is_table_continuation"] is True
    assert "Suite Tableau" in splitted[1].metadata["heading_refined"]