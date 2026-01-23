import os
import requests
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENVIRONMENT", "development")
os.environ["TOKENIZERS_PARALLELISM"] = "false" #To prevent a warning message, nothing really useful

def get_system_instruction_answer_generation():
    return """Tu es un assistant d’analyse documentaire.
Ton rôle est d’identifier, organiser et restituer fidèlement les informations présentes dans les documents fournis, sans simplification abusive ni interprétation externe.
Tu aides l’utilisateur à comprendre ce que disent les documents, et non à aller au-delà.

RÈGLES STRICTES À RESPECTER :
1. Réponds exclusivement en FRANÇAIS.
2. Base-toi UNIQUEMENT sur les informations présentes dans les DOCUMENTS fournis.
   - N'utilise JAMAIS tes connaissances personnelles ou externes.
   - Si une information n'est pas dans les documents, tu ne peux pas l'inventer ou la supposer.

3. Tu PEUX et DOIS :
   - Paraphraser et reformuler les informations des documents de manière naturelle et claire.
   - Combiner plusieurs passages des documents pour construire une réponse complète.
   - Faire des liens directs entre les informations explicitement présentes dans les documents.
   - Organiser et structurer les informations pour faciliter la compréhension.
   - Donner des exemples concrets quand ils sont présents dans les documents.
   - Identifier et lister des exemples lorsque les documents présentent explicitement plusieurs éléments appartenant à une même catégorie

4. Tu NE PEUX PAS :
   - Faire des inférences ou tirer des conclusions qui ne sont ni explicitement formulées, ni directement établies par des exemples donnés dans les documents.  
   - Ajouter des informations qui ne figurent pas dans les documents.
   - Utiliser tes connaissances générales sur le sujet.

5. PRÉCISION ET NUANCES :
   - Respecte les nuances et distinctions importantes du texte original.
   - Si un document fait une distinction précise (par exemple entre "les interdits se terminent" vs "l'état prend fin"), tu DOIS maintenir cette distinction.
   - Ne généralise pas de manière excessive - reste fidèle au niveau de détail du document.
   - Lorsque la réponse est une synthèse, ne jamais la présenter comme exhaustive ou normative.
   
6. GESTION DES RÉPONSES PARTIELLES :
   - Si tu trouves des informations partielles qui répondent à la question, fournis-les.
   - Sois honnête sur ce que tu présentes : distingue entre "tout ce qui est disponible" et "ce que je présente maintenant".
   - Si tu présentes un résumé mais que les documents contiennent plus de détails, dis : "Voici les points principaux. Les documents contiennent d'autres détails sur [mentionner les aspects non couverts]."
   - Si tu as vraiment tout couvert, dis : "C'est l'ensemble des informations pertinentes disponibles dans les documents sur cette question précise."
   - Exemple : Si on demande 5 étapes et que tu n'en trouves que 3, liste les 3 et ajoute "Les documents ne mentionnent que ces 3 étapes."

7. ABSENCE D'INFORMATION :
   - Si les documents ne contiennent AUCUNE information pertinente pour répondre à la question, réponds :
     "Je n'ai pas trouvé d'information dans les documents fournis qui permette de répondre à cette question."

8. TOLÉRANCE LINGUISTIQUE :
   - Sois flexible sur les variantes orthographiques (ex: Rusul/Rusl, Wudu/Woudou).
   - Accepte les synonymes courants (ex: "étape" = "pilier", "miqat" = "frontière") pour identifier l'information demandée.
   - Mais reformule toujours en utilisant les termes exacts des documents dans ta réponse.

PRINCIPE DIRECTEUR :
Ton objectif est d'être utile et pédagogique tout en restant 100% fidèle au contenu des documents. Aide l'utilisateur au maximum avec ce qui est disponible, mais ne franchis jamais la ligne en ajoutant des informations externes."""

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

def call_gpt_4o_mini(content_list, summarizing = False):
    """Function: call to openai's llm gpt-4o-mini, or gpt 4.1 nano, with the content_list parameter as a payload"""

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
    "temperature": 0,
    "max_tokens": 3000
    }   

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        res_json = response.json()
        return res_json["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Erreur API OpenAI : {e}")
        return "Désolé, je rencontre une difficulté technique pour générer la réponse."


def generate_answer_with_history(question, context_chunks,chat_history=None):
    """"Function : create an answer to the user {question}, by receiving {context_chunks} from the retriever/reranker."""
    if chat_history is None:
        chat_history = []

    # 1. Prepare the context from Qdrant
    formatted_contexts = []
    try:
        for i, chunk in enumerate(context_chunks):
            # On récupère les données structurées (text, tables, images)
            
            chunk_repr = f"--- EXTRAIT {i+1} ---\n"
            
            # Ajout du texte
            if chunk['text'] != "":
                chunk_repr += f"TEXTE: {chunk['text']}\n"
            
            # Ajout des tableaux (en HTML ou Markdown)
            if chunk['tables'] != []:
                for j, table in enumerate(chunk['tables']):
                    chunk_repr += f"\n[TABLEAU {j+1}]:\n{table}\n"

            #Les images sont dans un input séparé       
            formatted_contexts.append(chunk_repr)

        context_text = "\n\n".join(formatted_contexts)

        history_limit = int(os.getenv("CHAT_HISTORY_LIMIT", 6))
        history_str = ""
        # 2. Build the History string
        for msg in chat_history[-history_limit:]:
            history_str += f"{msg['role']}: {msg['content']}\n"


        # 3. Create the "Mega Prompt"
        prompt = f"""
        HISTORIQUE DES MESSAGES:
        {history_str}

        CONTEXTE EXTRAIT DU PDF:
        {context_text}

        QUESTION ACTUELLE: 
        {question}

        INSTRUCTIONS CRITIQUES:
        1. Réponds en utilisant UNIQUEMENT les informations des documents ci-dessus.
        2. Respecte les nuances et distinctions précises du texte original - ne simplifie pas excessivement.
        3. Donne des exemples concrets quand ils sont présents dans les documents.
        4. Si tu présentes un aperçu mais que les documents contiennent plus de détails, mentionne-le explicitement.
        5. Si aucune information pertinente n'est disponible, dis-le clairement.

        RÉPONSE:
        """
        content_list = [{"type": "text", "text": prompt}]

        for chunk in context_chunks: # Re picking every chunks to take into account images aswell
            for base64_image in chunk["images_base64"]:
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url" : f"data:image/jpeg;base64,{base64_image}"
                        }
                    
                })
        
        answer = call_gpt_4o_mini(content_list)

        # 5. Update History
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": str(answer)})


        return answer
    except Exception as e:
        print (f"Erreur lors de la génération de réponse... : {e}")
        return "Désolé, je rencontre une difficulté technique pour générer la réponse..."