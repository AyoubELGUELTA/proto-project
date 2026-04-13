# from app.indexing.operations.graph.graph_extractor import EntityAndRelationExtractor
# from app.indexing.operations.graph.summarize_manager import SummarizeManager
# from app.indexing.operations.entity_resolution.core_resolver import CoreResolver
# from app.indexing.operations.entity_resolution.llm_resolver import LLMResolver
# from app.indexing.operations.entity_resolution.resolution_engine import EntityResolutionEngine



# extractor=EntityAndRelationExtractor()

# # 1. On crée les resolvers
# core_res = CoreResolver(encyclopedia=encyclopedia_manager)
# llm_res = LLMResolver(llm_service=llm_service)

# # 2. On crée l'Engine
# engine = EntityResolutionEngine(core_resolver=core_res, llm_resolver=llm_res)

# # 3. On injecte l'Engine dans le GraphService
# graph_service = GraphService(
#     extractor=extractor,
#     summarize_extractor=summarize_extractor,
#     parser=parser,
#     resolution_engine=engine
# )