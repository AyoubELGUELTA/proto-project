import os
import requests
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENVIRONMENT", "development")


def get_system_instruction_answer_generation():
    return """Tu es un chercheur dont l'unique but est de vérifier si la QUESTION est contenue dans les DOCUMENTS reçus.

RÈGLES STRICTES :
1. Réponds exclusivement en FRANÇAIS.
2. Utilise UNIQUEMENT les documents fournis (toute connaissance personnelle ou externe est INTERDITE).
3. Si la réponse est présente dans les documents : 
   - Soit tu cites mot pour mot, soit tu combines plusieurs passages pour répondre. 
   - **Aucune interprétation, symbolisme ou ajout personnel n’est permis.**
4. Si les documents contiennent des informations partielles : 
   - Utilise uniquement ce qui est écrit, et précise que tu ne peux apporter plus d'information.
5. Si aucune information n’est trouvée : réponds "Je n'ai pas trouvé de réponse satisfaisante à cette question."

TOLÉRANCE PHONÉTIQUE & SÉMANTIQUE :
- Sois flexible sur l'orthographe (ex: Rusul/Rusl, Wudu/Woudou).
- Accepte les synonymes définis (ex: "étape" = "pilier", "miqat" = "frontière") uniquement pour matcher le texte.
- Ne transforme pas le sens des phrases originales."""


def call_gpt_4o_mini(content_list):
    """Function: call to openai's llm gpt-4o-mini with the content_list parameter as a payload"""

    api_key = os.getenv("OPENAI_API_KEY")

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }   
    payload = {
        "model": "gpt-4o-mini",
        "input": [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": get_system_instruction_answer_generation()}
                ]
            },
            {
                "role": "user",
                "content": 
                    content_list
                
            }
        ],
        "temperature": 0.1, # TO LOOK AT, perhaps lower it to 0 after several tests
        "max_output_tokens": 1000
    }

    response = requests.post(url, headers=headers, json=payload)
    res_json = response.json()

    try:
        return res_json["output"][0]["content"][0]["text"]
    except Exception:
        return f"Erreur API OpenAI : {res_json}"


# def call_prod_llm(prompt):#PROD
#     # SEE if i use GEMINI 2.5, OPENAI, other...

#     api_key = os.getenv("PROD_LLM_API_KEY") 
#     url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash-latest:generateContent?key={api_key}"    
#     payload = {"contents": [{"parts": [{"text": prompt}]}]}
#     response = requests.post(url, json=payload)
#     return response.json()['candidates'][0]['content']['parts'][0]['text']



def generate_answer_with_history(question, context_chunks,chat_history=None):
    """"Function : create an answer to the user {question}, by receiving {context_chunks} from the retriever/reranker."""
    # 1. Prepare the context from Qdrant
    formatted_contexts = []
    
    for i, chunk in enumerate(context_chunks):
        # On récupère les données structurées (text, tables, images)
        metadata = chunk.get('metadata', {})
        data = metadata.get('original_content', {}) # original content, which is text tables and images
        
        chunk_repr = f"--- EXTRAIT {i+1} ---\n"
        
        # Ajout du texte
        if data.get('raw_text'):
            chunk_repr += f"TEXTE: {data['raw_text']}\n"
        
        # Ajout des tableaux (en HTML ou Markdown)
        if data.get('tables_html'):
            for j, table in enumerate(data['tables_html']):
                chunk_repr += f"\n[TABLEAU {j+1}]:\n{table}\n"
        
        # we need to seperate the image in base64 input with the text input
        # if data.get('images_base64'):
        #     chunk_repr += f"\n(Note: Une image est présente dans cet extrait, décrite par le texte environnant.)\n" #TO CHANGE IN PROD
            
        formatted_contexts.append(chunk_repr)

    context_text = "\n\n".join(formatted_contexts)
    history_str = ""
    # 2. Build the History string
    if chat_history is not None:
        for msg in chat_history[-6:]:  # Keep only the last 6 messages (3 turns)
            history_str += f"{msg['role']}: {msg['content']}\n"
        print("Historique chat actuel:", history_str)  # Debug


    # 3. Create the "Mega Prompt"
    prompt = f"""
    HISTORIQUE DES MESSAGES:
    {history_str}

    CONTEXTE EXTRAIT DU PDF:
    {context_text}

    QUESTION ACTUELLE: 
    {question}

    RÉPONSE:
    """
    content_list = [{"type": "input_text", "text": prompt}]

    for chunk in context_chunks: # Re picking every chunks to take into account images aswell
        data = chunk.get("metadata", {}).get("original_content", {})
        for base64_image in data.get("images_base64", []):
            content_list.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
                
            })
    
    if ENV == "production":
        answer = call_gpt_4o_mini(content_list)
    else:
        answer = call_gpt_4o_mini(content_list)

    # 5. Update History
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": str(answer)})



    return answer