import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.json_parser = JsonOutputParser()
        
        # Compteurs cumulatifs pour la session
        self.total_tokens = 0
        self.total_cost = 0.0

    def reset_usage(self):
        """Réinitialise les compteurs de session."""
        self.total_tokens = 0
        self.total_cost = 0.0

    def get_session_report(self) -> str:
        """Retourne un rapport formaté de la consommation."""
        return (f"\n📊 --- RAPPORT DE CONSOMMATION LLM ---\n"
                f"🏷️ Modèle : {self.model_name}\n"
                f"🔢 Total Tokens : {self.total_tokens}\n"
                f"💰 Coût Total : ${self.total_cost:.4f}\n"
                f"--------------------------------------")

    async def generate_json(
        self, 
        input_data: Any = None, 
        system_prompt: str = None, 
        user_prompt: str = None, 
        show_usage: bool = False
    ) -> Dict[str, Any]:
        """Génère du JSON avec support du dispatching System/User et Caching."""
        try:
            messages = []
            
            # Cas 1 : Arguments nommés (utilisé dans ton script light)
            if system_prompt:
                sys_content = system_prompt
                if "JSON" not in sys_content:
                    sys_content += "\nRéponds exclusivement au format JSON."
                messages.append(SystemMessage(content=sys_content))
                messages.append(HumanMessage(content=user_prompt or ""))
            
            # Cas 2 : Dictionnaire (utilisé dans le refactor/post-process)
            elif isinstance(input_data, dict) and "system" in input_data:
                sys_content = input_data["system"]
                if "JSON" not in sys_content:
                    sys_content += "\nRéponds exclusivement au format JSON."
                messages.append(SystemMessage(content=sys_content))
                messages.append(HumanMessage(content=input_data.get("user", "")))
            
            # Cas 3 : Texte brut (Legacy)
            else:
                prompt_text = str(input_data)
                if "JSON" not in prompt_text:
                    prompt_text += "\nRéponds exclusivement au format JSON."
                messages.append(HumanMessage(content=prompt_text))

            # --- Appel API avec Monitoring ---
            with get_openai_callback() as cb:
                response = await self.llm.ainvoke(messages)
                parsed_response = self.json_parser.parse(response.content)
                
                self.total_tokens += cb.total_tokens
                self.total_cost += cb.total_cost
                
                if show_usage:
                    print(f"  [Token usage: {cb.total_tokens} | Cost: ${cb.total_cost:.5f}]")
                
                return parsed_response

        except Exception as e:
            logger.error(f"❌ Erreur LLM JSON: {e}")
            return {}
        
    async def generate_text(self, prompt_text: str, show_usage: bool = False) -> str:
        """Génère du texte brut et suit la consommation."""
        with get_openai_callback() as cb:
            response = await self.llm.ainvoke(prompt_text)
            
            self.total_tokens += cb.total_tokens
            self.total_cost += cb.total_cost
            
            if show_usage:
                print(f"  [Token usage: {cb.total_tokens} | Cost: ${cb.total_cost:.5f}]")
                
            return response.content