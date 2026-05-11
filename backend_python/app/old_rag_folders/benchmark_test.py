

# def get_benchmark_config_rag(config_id: str):
#     configs = {
#         "01": {"top_k": 30, "top_n": 15, "prompt_style": "light"},
#         "02": {"top_k": 30, "top_n": 15, "prompt_style": "verbose"},
#         "03": {"top_k": 50, "top_n": 15, "prompt_style": "light"},
#         "04": {"top_k": 50, "top_n": 20, "prompt_style": "light"},
#         "05": {"top_k": 30, "top_n": 15, "prompt_style": "verbose"},
#         "06": {"top_k": 80, "top_n": 13, "prompt_style": "verbose"},
#         "07": {"top_k": 50, "top_n": 15, "prompt_style": "reasoning"},
#         "08": {"top_k": 80, "top_n": 13, "prompt_style": "verbose"},
#         "09": {"top_k": 40, "top_n": 15, "prompt_style": "light"},
#         "10": {"top_k": 50, "top_n": 15, "prompt_style": "reasoning"},
#         "11": {"top_k": 60, "top_n": 15, "prompt_style": "verbose"}

#     }
#     return configs.get(config_id, configs["01"])


# def get_ingest_benchmark_config(config_id: str):
#     """
#     Retourne les paramètres de chunking selon l'ID du test benchmark.
#     """
#     configs = {
#         # Tests basés sur la structure automatique de Docling
#         "01": {"mode": "docling_auto", "chunk_size": 1200, "overlap": 150},
#         "02": {"mode": "docling_auto", "chunk_size": 3750, "overlap": 250},
#         "03": {"mode": "docling_auto", "chunk_size": 3750, "overlap": 250},
#         "08": {"mode": "docling_auto", "chunk_size": 3750, "overlap": 250},
#         "07": {"mode": "docling_auto", "chunk_size": 3750, "overlap": 250},

        
#         # Tests basés sur le découpage récursif (RecursiveCharacterTextSplitter)
#         "04": {"mode": "recursive", "chunk_size": 1000, "overlap": 100}, # 10% overlap
#         "05": {"mode": "recursive", "chunk_size": 1500, "overlap": 150},
#         "09": {"mode": "recursive", "chunk_size": 1000, "overlap": 100},
#         "10": {"mode": "recursive", "chunk_size": 1500, "overlap": 150},
#         "11": {"mode": "recursive", "chunk_size": 1500, "overlap": 150},


#         # Test "Super Chunks" (limités pour éviter la saturation du contexte)
#         "06": {"mode": "recursive", "chunk_size": 2500, "overlap": 250},
#             }
    
#     # Fallback par défaut sur Docling Auto si l'ID est inconnu
#     return configs.get(config_id, configs["01"])