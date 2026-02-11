from .rag.query_rewriter import rewrite_query
from .rag.answer_generator import generate_answer_with_history
from .rag.retriever import retrieve_chunks

async def run_benchmark_test(config_id, question, chat_history=[]):
    # Dictionnaire des configurations basées sur notre tableau
    configs = {
        "01": {"chunk_type": "docling", "top_k": 30, "top_n": 15, "prompt_style": "light"},
        "02": {"chunk_type": "docling", "top_k": 30, "top_n": 15, "prompt_style": "verbose"},
        "03": {"chunk_type": "docling", "top_k": 50, "top_n": 15, "prompt_style": "light"},
        "04": {"chunk_type": "recursive_1000", "top_k": 65, "top_n": 20, "prompt_style": "light"},
        "05": {"chunk_type": "recursive_1500", "top_k": 40, "top_n": 15, "prompt_style": "verbose"},
        "06": {"chunk_type": "recursive_2500", "top_k": 80, "top_n": 20, "prompt_style": "verbose"},
        "07": {"chunk_type": "docling", "top_k": 50, "top_n": 15, "prompt_style": "reasoning"},
    }

    c = configs.get(config_id)
    if not c: return "Config ID inconnue"

    # 1. On appelle le rewriter (V1, V2, V3 + KW)
    query_data = await rewrite_query(question, chat_history)

    # 2. Retrieval avec les paramètres de la config
    # On passe top_k pour le retrieval et top_n pour le reranker
    context_chunks = await retrieve_chunks(
        query_data, 
        limit=c["top_k"], 
        rerank_limit=c["top_n"]
    )

    # 3. Génération de la réponse avec le style de prompt choisi
    response = await generate_answer_with_history(
        question, 
        context_chunks, 
        style=c["prompt_style"]
    )

    return response

def get_benchmark_config_rag(config_id: str):
    configs = {
        "01": {"top_k": 30, "top_n": 15, "prompt_style": "light"},
        "02": {"top_k": 30, "top_n": 15, "prompt_style": "verbose"},
        "03": {"top_k": 50, "top_n": 15, "prompt_style": "light"},
        "04": {"top_k": 50, "top_n": 20, "prompt_style": "light"},
        "05": {"top_k": 30, "top_n": 15, "prompt_style": "verbose"},
        "06": {"top_k": 80, "top_n": 13, "prompt_style": "verbose"},
        "07": {"top_k": 50, "top_n": 15, "prompt_style": "reasoning"},
        "08": {"top_k": 80, "top_n": 13, "prompt_style": "verbose"},
        "09": {"top_k": 40, "top_n": 15, "prompt_style": "light"},
        "10": {"top_k": 50, "top_n": 15, "prompt_style": "reasoning"},
        "11": {"top_k": 60, "top_n": 15, "prompt_style": "verbose"}

    }
    return configs.get(config_id, configs["01"])


def get_ingest_benchmark_config(config_id: str):
    """
    Retourne les paramètres de chunking selon l'ID du test benchmark.
    """
    configs = {
        # Tests basés sur la structure automatique de Docling
        "01": {"mode": "docling_auto", "chunk_size": None, "overlap": None},
        "02": {"mode": "docling_auto", "chunk_size": None, "overlap": None},
        "03": {"mode": "docling_auto", "chunk_size": None, "overlap": None},
        "08": {"mode": "docling_auto", "chunk_size": None, "overlap": None},
        "07": {"mode": "docling_auto", "chunk_size": None, "overlap": None},

        
        # Tests basés sur le découpage récursif (RecursiveCharacterTextSplitter)
        "04": {"mode": "recursive", "chunk_size": 1000, "overlap": 100}, # 10% overlap
        "05": {"mode": "recursive", "chunk_size": 1500, "overlap": 150},
        "09": {"mode": "recursive", "chunk_size": 1000, "overlap": 100},
        "10": {"mode": "recursive", "chunk_size": 1500, "overlap": 150},
        "11": {"mode": "recursive", "chunk_size": 1500, "overlap": 150},


        # Test "Super Chunks" (limités pour éviter la saturation du contexte)
        "06": {"mode": "recursive", "chunk_size": 2500, "overlap": 250},
            }
    
    # Fallback par défaut sur Docling Auto si l'ID est inconnu
    return configs.get(config_id, configs["01"])