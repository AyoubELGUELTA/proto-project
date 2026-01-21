import os
from .answer_generator import call_gpt_4o_mini



def rewrite_query(latest_question, chat_history):
    """"Rewrite the latest_question of the user by taking into account the chat_history 
    to make a standalone question for the retriever"""
    
    if not chat_history:
        return latest_question
    
    print (chat_history)
    
    try:
        # On limite l'historique (configurable via env pour économiser des tokens)
        history_limit = int(os.getenv("CHAT_HISTORY_LIMIT", 6))
        
        history_str = ""
        for msg in chat_history[-history_limit:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            history_str += f"{role}: {content}\n"
    

            prompt = f"""
        ROLE : PRENDS EN COMPTE LES INSTRUCTIONS SYSTEMES QUE TU AS REÇUES.

        Historique de la conversation:
        {history_str}

        Dernière question:
        {latest_question}
        """

        content_list = [{"type": "text", "text": prompt}]

        rewritten_query = call_gpt_4o_mini(content_list, summarizing= True)

        # Sécurité : Si le LLM renvoie n'importe quoi ou vide
        if not rewritten_query or len(rewritten_query.strip()) < 2:
            return latest_question
            
        return rewritten_query.strip()
    
    except Exception as e:
        print(f"⚠️ Échec du Query Rewriting: {e}. Utilisation de la question originale.")
        # Fallback critique : si l'IA de réécriture plante, 
        # on utilise la question brute pour ne pas bloquer le RAG
        return latest_question
