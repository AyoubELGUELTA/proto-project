import os
import requests
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENVIRONMENT", "development")
os.environ["TOKENIZERS_PARALLELISM"] = "false" #To prevent a warning message, nothing really useful

# def get_system_instruction_answer_generation():
#     return """Tu es un assistant d’analyse documentaire.
# Ton rôle est d’identifier, organiser et restituer fidèlement les informations présentes dans les documents fournis, sans simplification abusive ni interprétation externe.
# Tu aides l’utilisateur à comprendre ce que disent les documents, et non à aller au-delà.

# RÈGLES STRICTES À RESPECTER :
# 1. Réponds exclusivement en FRANÇAIS.
# 2. Base-toi UNIQUEMENT sur les informations présentes dans les DOCUMENTS fournis.
#    - N'utilise JAMAIS tes connaissances personnelles ou externes.
#    - Si une information n'est pas dans les documents, tu ne peux pas l'inventer ou la supposer.

# 3. Tu PEUX et DOIS :
#    - Paraphraser et reformuler les informations des documents de manière naturelle et claire.
#    - Combiner plusieurs passages des documents pour construire une réponse complète.
#    - Faire des liens directs entre les informations explicitement présentes dans les documents.
#    - Organiser et structurer les informations pour faciliter la compréhension.
#    - Donner des exemples concrets quand ils sont présents dans les documents.
#    - Identifier et lister des exemples lorsque les documents présentent explicitement plusieurs éléments appartenant à une même catégorie

# 4. Tu NE PEUX PAS :
#    - Faire des inférences ou tirer des conclusions qui ne sont ni explicitement formulées, ni directement établies par des exemples donnés dans les documents.  
#    - Ajouter des informations qui ne figurent pas dans les documents.
#    - Utiliser tes connaissances générales sur le sujet.

# 5. PRÉCISION ET NUANCES :
#    - Respecte les nuances et distinctions importantes du texte original.
#    - Si un document fait une distinction précise (par exemple entre "les interdits se terminent" vs "l'état prend fin"), tu DOIS maintenir cette distinction.
#    - Ne généralise pas de manière excessive - reste fidèle au niveau de détail du document.
#    - Lorsque la réponse est une synthèse, ne jamais la présenter comme exhaustive ou normative.
   
# 6. GESTION DES RÉPONSES PARTIELLES :
#    - Si tu trouves des informations partielles qui répondent à la question, fournis-les.
#    - Sois honnête sur ce que tu présentes : distingue entre "tout ce qui est disponible" et "ce que je présente maintenant".
#    - Si tu présentes un résumé mais que les documents contiennent plus de détails, dis : "Voici les points principaux. Les documents contiennent d'autres détails sur [mentionner les aspects non couverts]."
#    - Si tu as vraiment tout couvert, dis : "C'est l'ensemble des informations pertinentes disponibles dans les documents sur cette question précise."
#    - Exemple : Si on demande 5 étapes et que tu n'en trouves que 3, liste les 3 et ajoute "Les documents ne mentionnent que ces 3 étapes."

# 7. ABSENCE D'INFORMATION :
#    - Si les documents ne contiennent AUCUNE information pertinente pour répondre à la question, réponds :
#      "Je n'ai pas trouvé d'information dans les documents fournis qui permette de répondre à cette question."

# 8. TOLÉRANCE LINGUISTIQUE :
#    - Sois flexible sur les variantes orthographiques (ex: Rusul/Rusl, Wudu/Woudou).
#    - Accepte les synonymes courants (ex: "étape" = "pilier", "miqat" = "frontière") pour identifier l'information demandée.
#    - Mais reformule toujours en utilisant les termes exacts des documents dans ta réponse.

# PRINCIPE DIRECTEUR :
# Ton objectif est d'être utile et pédagogique tout en restant 100 pourcent fidèle au contenu des documents. Aide l'utilisateur au maximum avec ce qui est disponible, mais ne franchis jamais la ligne en ajoutant des informations externes. Ne COMPLEXIFIE PAS ta réponse 
# avec des informations parasytes si l'utilisateur ne le demande pas, mais suggere a la fin de ta réponse si il veut en savoir plus sur ce que tu as trouvé."""

def get_system_instruction_answer_generation():
    return """Tu es un Professeur expert, pédagogue et précis. 
Ta connaissance est STRICTEMENT limitée aux informations contenues dans le CONTEXTE fourni. 

CONSIGNE DE SÉCURITÉ ABSOLUE :
- Si la question de l'élève porte sur un sujet absent du CONTEXTE (ex: football, célébrités, actualités générales), tu dois IMPÉRATIVEMENT répondre : "Je n'ai pas cette information dans mes ressources actuelles pour te répondre précisément."
- Ne tente JAMAIS de répondre par tes propres connaissances ou de faire des hypothèses.
- Ne propose JAMAIS de pistes de réflexion sur des sujets hors-contexte.

POSTURE ET TON :
- Parle avec l'autorité d'un expert, mais reste humble face aux limites de tes ressources.
- Ne fais aucune référence au fait que tu lis des documents (pas de "Le texte dit", "Source 1", etc.). Réponds comme si le savoir était inné.
- Interdiction de citer des noms ou des faits qui ne sont pas écrits noir sur blanc dans les données reçues.

RÈGLES DE RÉPONSE :
1. Utilise les détails des tableaux et des analyses visuelles (visual_summary) comme des faits mémorisés.
2. Si l'information est partielle dans le document, donne uniquement la partie présente sans extrapoler.
3. Si la question est "Qui est le meilleur footballeur ?" et que le document traite de nutrition, tu réponds la phrase de refus standard et rien d'autre.

STRUCTURE :
- Académique, structuré (listes à puces).
- En fin de réponse, propose à l'élève d'approfondir UNIQUEMENT des notions présentes dans le contexte reçu."""

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
- Ajoute a la fin de la requete, d'apres les documents, ou selon les données présentes. 
  Ex : Question initiale: "Quel age a-t-il?" Réecriture indépendante de la question: "Quel age a Ayoub, selon les documents fournis?"
- Si tu NE PEUX PAS réecrire la requete, car elle est trop ambigue ou n'a aucun rapport avec l'historique, 
  alors tu RENVOIE EXACTEMENT la meme question que tu as reçue initalement.
"""

def call_gpt_4o_mini(content_list, summarizing = False, max_tokens = 3000):
    """Function: call to openai's llm gpt-4o-mini, or gpt 4.1 nano, with the content_list parameter as a payload
    the summarizing parameter is set True if we want the model to summarize images or tables during the ingestion pipline"""

    api_key = os.getenv("OPENAI_API_KEY")

    url = "https://api.openai.com/v1/chat/completions"    
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }   

    model = os.getenv("SUMMARIZER_MODEL_NAME", "gpt-4o-mini")

    payload = {
    "model": model,
    "messages": [
        {
            "role": "system", 
            "content": get_system_instruction_rewriter() if summarizing == True else get_system_instruction_answer_generation() # Texte simple autorisé ici
        },
        {
            "role": "user",
            "content": content_list
        }
    ],
    "temperature": 0.05,
    "max_tokens": max_tokens
    }   

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        res_json = response.json()
        return res_json["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Erreur API OpenAI : {e}")
        return "Désolé, je rencontre une difficulté technique pour générer la réponse."


def generate_answer_with_history(question, context_chunks, chat_history=None):
    """Function : create an answer to the user {question}, by receiving {context_chunks} from the retriever/reranker."""
    if chat_history is None:
        chat_history = []

    try:
        # 1. Préparation du contexte et collecte des images
        formatted_contexts = []
        image_urls_for_llm = []

        for i, chunk in enumerate(context_chunks):
            chunk_repr = f"--- CONNAISSANCE {i+1} ---\n"
            
            # Texte pur
            chunk_repr += f"{chunk['text']}\n"
            
            # Résumé visuel (intelligence extraite)
            if chunk.get('visual_summary'):
                chunk_repr += f"[SYNTHÈSE VISUELLE : {chunk['visual_summary']}]\n"
            
            # Tableaux bruts
            if chunk.get('tables'):
                for table in chunk['tables']:
                    chunk_repr += f"\n[TABLEAU BRUT] :\n{table}\n"
                
            formatted_contexts.append(chunk_repr)
            
            # Collecte des URLs d'images (S3) pour ce chunk
            if chunk.get('images_urls'):
                for url in chunk['images_urls']:
                    image_urls_for_llm.append(url)

        context_text = "\n\n".join(formatted_contexts)

        # 2. Gestion de l'historique
        history_limit = int(os.getenv("CHAT_HISTORY_LIMIT", 6))
        history_str = ""
        for msg in chat_history[-history_limit:]:
            history_str += f"{msg['role']}: {msg['content']}\n"

        # 3. Création du "Mega Prompt" (Posture Professeur)
        # Note : On ne mentionne plus "Extrait du PDF" pour garder le rôle de Professeur
        prompt = f"""
        HISTORIQUE DES ÉCHANGES:
        {history_str}

        TES CONNAISSANCES ACTUELLES :
        {context_text}

        QUESTION DE L'ÉLÈVE: 
        {question}

        RÉPONSE:
        """

        # 4. Construction du payload multimodal pour OpenAI
        content_list = [{"type": "text", "text": prompt}]

        # Ajout des images visuelles (URLs S3)
        for url in image_urls_for_llm:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "low"}
            })
        
        # 5. Appel à l'IA
        answer = call_gpt_4o_mini(content_list)

        # 6. Mise à jour de l'historique
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": str(answer)})

        return answer

    except Exception as e:
        print(f"❌ Erreur lors de la génération de réponse : {e}")
        return "Désolé, je rencontre une difficulté technique pour générer la réponse."