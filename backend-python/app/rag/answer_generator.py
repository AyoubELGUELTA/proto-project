import os
import httpx 
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

def get_prompt_light():
    return """Vous êtes un Professeur expert. Répondez à l'élève de manière directe.
- Votre connaissance est strictement limitée au contexte fourni.
- Si le sujet est absent, répondez : "Je ne peux pas aider sur ce sujet."
- Pas de verbes de citation ("le document dit").
- Structurez en listes ou paragraphes clairs, sans introduction."""

def get_prompt_verbose():
    return """Tu es un Professeur expert, analyse documentaire.
Ton rôle est d’identifier, organiser et restituer fidèlement les informations des documents, sans simplification abusive, et de répondre à l'élève de manière directe.

RÈGLES :
1. BASE-TOI UNIQUEMENT sur les documents fournis. N'utilise JAMAIS tes connaissances externes.
2. NUANCES : Respecte les distinctions précises (ex: "l'état prend fin" vs "les interdits se terminent").
3. RÉPONSES PARTIELLES : Si tu n'as que 3 étapes sur 5 demandées, liste les 3 et précise : "Voici l'ensemble des informations qui sont a ma connaissance." ou une phrase du genre.
4. TOLÉRANCE LINGUISTIQUE : Accepte les variantes (Wudu/Woudou) mais utilise les termes exacts du texte dans ta réponse.
5. HONNÊTETÉ : Si l'info est absente, dis : "Je n'ai pas trouvé d'information dans les documents fournis." ou une phrase du style.
6. STRUCTURE : Aide l'utilisateur avec des exemples concrets extraits des documents.
En fin de réponse, suggère un approfondissement basé sur ce que tu as trouvé dans les connaissances reçues."""

def get_prompt_reasoning():
    return """Tu es un analyste rigoureux. Avant de répondre, tu dois décomposer ton raisonnement.

STRUCTURE IMPÉRATIVE :
1. <pensee> : 
   - Liste les entités (noms propres, lieux) trouvées dans les chunks.
   - Identifie les dates ou la chronologie.
   - Note les éventuelles contradictions entre les sources.
</pensee>

2. RÉPONSE FINALE : 
   - Applique les règles de fidélité absolue (PAS DE CONNAISSANCES EXTERNES).
   - Réponds de manière structurée et pédagogique.
   - Si une information manque pour conclure, mentionne-le explicitement.
   - En fin de réponse, suggère un approfondissement basé sur ce que tu as trouvé dans les connaissances reçues."""

def get_system_instruction_rewriter():
    return """
    Tu es un expert en ingénierie de la connaissance islamique et en optimisation de recherche RAG.
    Ta mission : Transformer la requête utilisateur en vecteurs de recherche exploratoires sans jamais présumer de la réponse.

    AXES DE RÉDACTION :
    1. LINGUISTIQUE : Utilise systématiquement la terminologie bilingue (ex: Prière/Salat, Unicité/Tawhid).
    2. CONTEXTUEL : Ajoute des termes de domaines académiques (Fiqh, Sira, Aqida, Hadith) pour cibler les bons chapitres.
    3. STRUCTUREL : Si la question demande une liste ou un "qui", utilise des termes décrivant la catégorie d'appartenance (ex: "membres de la famille", "compagnons mentionnés") SANS lister de noms spécifiques que tu connais déjà.

    CONSIGNES DES VARIANTES :
    - V1 (Fidélité & Historique) : Reformulation claire résolvant les pronoms (ex: remplace "il" par le sujet précédent en te basant sur l'historique).
    - V2 (Contexte & Domaine) : Oriente la recherche vers le cadre juridique, historique ou spirituel du sujet.
    - V3 (Variantes Techniques) : Concentre-toi sur les termes techniques arabes et leurs transcriptions phonétiques variées.
    - KEYWORDS : Liste brute de noms propres et termes techniques. Pour chaque terme arabe, propose 2-3 variantes orthographiques (ex: 'Aisha, Aicha, Ayesha).

    INTERDICTION :
    - Ne jamais inclure d'exemples de réponses (ex: si on demande les noms des compagnons, ne cite pas "Abu Bakr" dans tes variantes).

    FORMAT DE SORTIE :
    V1: [Phrase]
    V2: [Phrase]
    V3: [Phrase]
    KEYWORDS: [Mot1, Mot2, Mot3...]"""

async def call_gpt_4o_mini(content_list, rewriting=False, style="verbose", max_tokens=3000):
    """
    Appel à l'IA avec sélection dynamique du prompt système.
    Styles disponibles : 'light', 'verbose', 'reasoning'
    """
    api_key = os.getenv("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/chat/completions"    
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }   

    # Sélection du prompt système
    if rewriting:
        system_instruction = get_system_instruction_rewriter()
    else:
        # Mapping des styles pour le benchmark
        prompts = {
            "light": get_prompt_light(),
            "verbose": get_prompt_verbose(),
            "reasoning": get_prompt_reasoning()
        }
        system_instruction = prompts.get(style, get_prompt_verbose())

    model = os.getenv("SUMMARIZER_MODEL_NAME", "gpt-4o-mini")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": content_list}
        ],
        "temperature": 0.05 if rewriting else 0.25,
        "max_tokens": max_tokens
    }   

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=90.0)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Erreur API OpenAI : {e}")
        return "Désolé, je rencontre une difficulté technique pour générer la réponse."


async def generate_answer_with_history(question, context_chunks, chat_history=None, style="verbose"):
    """
    Génère une réponse en utilisant le formateur de contexte enrichi 
    et en gérant le payload multimodal (images S3).
    """
    if chat_history is None:
        chat_history = []

    try:
        # 1. Préparation du contexte textuel et collecte des images
        formatted_parts = []
        image_urls_for_llm = []
        
        # On garde une trace des URLs pour ne pas envoyer 10 fois la même image
        seen_image_urls = set()

        for chunk in context_chunks:
            # Gestion de l'Identité du Document (si présente via ton retriever)
            if chunk.get("is_identity"):
                title = chunk.get('title', 'Document sans titre')
                formatted_parts.append(f"\n===== SOURCE : {title} =====")
                continue

            # Construction du bloc de connaissance précis
            idx = chunk.get('chunk_index', '?')
            pages = chunk.get('page_numbers', [])
            page_str = f"Page(s): {', '.join(map(str, pages))}" if pages else "Page: N/A"
            
            # On privilégie text_for_reranker car il contient déjà souvent le heading + visual_summary
            text_content = chunk.get('text_for_reranker', chunk.get('text', ''))
            
            chunk_repr = f"\n[CONNAISSANCE #{idx} | {page_str}]\n{text_content}\n"
            
            # Ajout des tableaux s'ils ne sont pas déjà dans le text_for_reranker
            if chunk.get('tables'):
                for table in chunk['tables']:
                    chunk_repr += f"\n[DONNÉES TABLEAU] :\n{table}\n"
            
            formatted_parts.append(chunk_repr)
            
            # Collecte des images pour le payload multimodal
            if chunk.get('images_urls'):
                for url in chunk['images_urls']:
                    if url not in seen_image_urls:
                        image_urls_for_llm.append(url)
                        seen_image_urls.add(url)

        context_text = "\n".join(formatted_parts)

        # 2. Gestion de l'historique (limite définie dans tes env)
        history_limit = int(os.getenv("CHAT_HISTORY_LIMIT", 6))
        history_str = ""
        for msg in chat_history[-history_limit:]:
            role = "Élève" if msg['role'] == 'user' else "Professeur"
            history_str += f"{role}: {msg['content']}\n"

        # 3. Création du Prompt final
        prompt = f"""
        HISTORIQUE DES ÉCHANGES:
        {history_str}

        TES CONNAISSANCES ACTUELLES:
        {context_text}

        QUESTION DE L'ÉLÈVE: 
        {question}

        RÉPONSE DU PROFESSEUR:
        """

        # 4. Construction du payload multimodal pour OpenAI
        # Le texte contient toutes les instructions et le contexte
        content_list = [{"type": "text", "text": prompt}]

        # On ajoute les images physiques à la fin pour que le LLM puisse les "voir"
        for url in image_urls_for_llm:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "low"}
            })
        
        # 5. Appel à l'IA
        answer = await call_gpt_4o_mini(content_list, style=style)

        # 6. Mise à jour de l'historique
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": str(answer)})

        return answer

    except Exception as e:
        print(f"❌ Erreur lors de la génération de réponse : {e}")
        return "Désolé, je rencontre une difficulté technique pour générer la réponse."