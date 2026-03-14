# #FILE TODO if results are more pertinent in tests incoming
# async def detect_duplicates():
#     """
#     Détecte doublons potentiels avec scoring multi-niveaux.
#     """
    
#     entities = await fetch_all_entities()
    
#     duplicates = []
    
#     for i, entity_a in enumerate(entities):
#         for entity_b in entities[i+1:]:
            
#             # RÈGLE 1 : Inclusion stricte
#             # "Abu Talib" ⊆ "Abu Talib ibn Abd al-Muttalib"
#             if is_substring(entity_a, entity_b):
#                 duplicates.append({
#                     'entity_a': entity_a,
#                     'entity_b': entity_b,
#                     'confidence': 0.95,
#                     'reason': 'substring_exact',
#                     'action': 'auto_merge'  # ← Merge automatique
#                 })
#                 continue
            
#             # RÈGLE 2 : Similarité Levenshtein élevée
#             # "Muhammad ibn Abdullah" vs "Mohammed ibn Abdallah"
#             sim = levenshtein_similarity(entity_a['normalized_name'], entity_b['normalized_name'])
#             if sim > 0.85:
#                 duplicates.append({
#                     'entity_a': entity_a,
#                     'entity_b': entity_b,
#                     'confidence': sim,
#                     'reason': 'high_similarity',
#                     'action': 'llm_confirm'  # ← Demande au LLM
#                 })
#                 continue
            
#             # RÈGLE 3 : Alias overlap
#             # entity_a a un alias qui match entity_b
#             common_aliases = set(entity_a['normalized_aliases']) & set(entity_b['normalized_aliases'])
#             if len(common_aliases) >= 2:
#                 duplicates.append({
#                     'entity_a': entity_a,
#                     'entity_b': entity_b,
#                     'confidence': 0.75,
#                     'reason': f'shared_aliases: {common_aliases}',
#                     'action': 'llm_confirm'
#                 })
#                 continue
            
#             # RÈGLE 4 : Prénom + patronyme partiel
#             # "Ibrahim ibn Muhammad" vs "Ibrahim"
#             if is_short_form(entity_a, entity_b):
#                 duplicates.append({
#                     'entity_a': entity_a,
#                     'entity_b': entity_b,
#                     'confidence': 0.60,
#                     'reason': 'short_form',
#                     'action': 'llm_confirm'
#                 })
    
#     return duplicates