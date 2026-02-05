import os
from .answer_generator import call_gpt_4o_mini



def rewrite_query(latest_question, chat_history):
    if not chat_history:
        # On ajoute quand même le petit suffixe "selon les documents" défini dans ton système
        return latest_question
    
    try:
        history_limit = int(os.getenv("CHAT_HISTORY_LIMIT", 10))
        
        # 1. On construit l'historique COMPLET (User + Assistant)
        history_str = ""
        for msg in chat_history[-history_limit:]:
            role = "Élève" if msg.get('role') == 'user' else "Professeur"
            content = msg.get('content', '')
            history_str += f"{role}: {content}\n"
    
        prompt = f"""
        HISTORIQUE DE LA DISCUSSION :
        {history_str}

        DERNIÈRE QUESTION DE L'ÉLÈVE :
        {latest_question}

        CONSIGNE : 
        En utilisant l'historique, reformule la dernière question pour qu'elle soit compréhensible sans contexte. 
        Respecte strictement les instructions système (langue française, ajout de "selon les documents").

        """

        content_list = [{"type": "text", "text": prompt}]

        # On appelle le LLM une seule fois, après avoir construit history_str
        rewritten_query = call_gpt_4o_mini(content_list, summarizing=True)

        if not rewritten_query or len(rewritten_query.strip()) < 2:
            return latest_question
            
        print ("REFORMULATION DE LA QUERY :", rewritten_query.strip())
        return rewritten_query.strip()
    
    except Exception as e:
        print(f"⚠️ Échec du Query Rewriting: {e}")
        return latest_question
