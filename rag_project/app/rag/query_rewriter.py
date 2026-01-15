from .answer_generator import call_gpt_4o_mini

def get_system_instruction_rewriter():
    return """Tu es un réécrivain de requêtes.
Ta SEULE tâche consiste à réécrire la dernière question de l'utilisateur
en une question autonome et entièrement explicite, EN FRANCAIS.

Règles :
- Utilise UNIQUEMENT l'historique de la conversation.
- Ne réponds PAS à la question.
- N'ajoute PAS de nouvelles informations.
- Ne fais PAS de déductions.
- Si l'intention est ambiguë, conserve l'ambiguïté.
- Ne produis QUE la requête réécrite.
"""

def rewrite_query(latest_question, chat_history):
    """"Rewrite the latest_question of the user by taking into account the chat_history 
    to make a standalone question for the retriever"""
    history_str = ""
    for msg in chat_history[-6:]:
        history_str += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
{get_system_instruction_rewriter()}

Historique de la conversation:
{history_str}

Dernière question:
{latest_question}
"""

    content_list = [{"type": "input_text", "text": prompt}]
    return call_gpt_4o_mini(content_list)
