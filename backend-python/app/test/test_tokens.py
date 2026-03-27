import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from app.ingestion.graph_prompts import SIRA_ENTITY_P1_SYSTEM, SIRA_RELATION_P1_SYSTEM
load_dotenv()



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_step(step_name, system_prompt, user_content):
    print(f"\n🚀 TEST : {step_name}")
    for i in range(2):  # On passe 2 fois pour voir le cache s'activer
        print(f"  Appel {i+1}...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"} # Force le JSON
        )
        
        usage = response.usage
        cached = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
        
        print(f"    - Tokens d'entrée : {usage.prompt_tokens}")
        print(f"    - Tokens en CACHE : {cached} {'✅ (Économie !)' if cached > 0 else '⏳ (Premier passage)'}")
        
        if i == 1: # On affiche le résultat seulement au 2ème passage
            print(f"    - Résultat JSON : {response.choices[0].message.content[:200]}...")

# --- DONNÉES DE TEST ---
TEXT_SAMPLE = """
Hamza ibn Abdul-Muttalib RA, l'oncle du Prophète ﷺ, a fait preuve d'une bravoure 
immense lors de la bataille de Badr. Il était accompagné de Ali ibn Abi Talib RA.
"""

if __name__ == "__main__":
    # 1. Test des Entités
    test_step("EXTRACTION ENTITÉS (P1)", SIRA_ENTITY_P1_SYSTEM, TEXT_SAMPLE)
    
    # 2. Test des Relations (Simulé avec un exemple d'entités)
    RELATION_INPUT = "Entités trouvées: [Hamza, Ali, Badr]. Extrais les liens."
    test_step("EXTRACTION RELATIONS (P1)", SIRA_RELATION_P1_SYSTEM, RELATION_INPUT)