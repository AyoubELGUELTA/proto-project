import os
from .answer_generator import call_gpt_4o_mini



async def rewrite_query(latest_question, chat_history):
    
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
        """

        content_list = [{"type": "text", "text": prompt}]

        # On appelle le LLM une seule fois, après avoir construit history_str
        try:
            raw_output = await call_gpt_4o_mini(content_list, rewriting=True)
            
            # Parsing robuste
            lines = raw_output.strip().split('\n')
            v1, v2, v3, keywords = "", "", "", ""
            
            for line in lines:
                if line.startswith("V1:"): v1 = line.replace("V1:", "").strip()
                elif line.startswith("V2:"): v2 = line.replace("V2:", "").strip()
                elif line.startswith("V3:"): v3 = line.replace("V3:", "").strip()
                elif line.startswith("KEYWORDS:"): keywords = line.replace("KEYWORDS:", "").strip()

            # Fallback si le parsing échoue
            if not v1: v1 = latest_question

            print(f"🔎 MULTI-QUERY REWRITER ACTIVÉ")
            print(f"   V1: {v1}\n   V2: {v2}\n   V3: {v3}\n   KW: {keywords}")

            return {
                "vector_query": v1, # On garde V1 comme référence principale (pour le reranker)
                "variants": [v1, v2, v3],
                "keyword_query": keywords
            }

        except Exception as e:
            print(f"⚠️ Échec du Query Rewriting: {e}")
            return {"vector_query": latest_question, "variants": [latest_question], "keyword_query": latest_question}
    
    except Exception as e:
        print(f"⚠️ Échec du Query Rewriting: {e}")
        return {"vector_query": latest_question, "variants": [latest_question], "keyword_query": latest_question}