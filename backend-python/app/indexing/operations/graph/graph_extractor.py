# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import logging
import pandas as pd
from typing import List, Tuple, Dict, Any

from app.core.config.graph_config import ExtractionConfig 
from app.services.llm.client import LLMClient
from app.services.llm.parser import LLMParser
from app.core.prompts.graph_prompts import GRAPH_EXTRACTION_PROMPT, CONTINUE_PROMPT, LOOP_PROMPT

logger = logging.getLogger(__name__)

class GraphExtractor:
    def __init__(
        self,
        llm_client: LLMClient,
        config: ExtractionConfig
    ):
        self.llm = llm_client
        self.config = config
        self.parser = LLMParser()

    async def _extract_with_gleaning(self, text: str, context: str) -> str:
        """Boucle de Gleaning utilisant la config injectée."""
        
        prompt_input = GRAPH_EXTRACTION_PROMPT.format(
            entity_types=",".join(self.config.entity_types),
            document_metadata=context,
            input_text=text
        )

        messages = [{"role": "user", "content": prompt_input}]
        
        # Première passe
        response = await self.llm.ask(messages)
        full_results = response
        
        # Gleaning logic
        if self.config.max_gleanings > 0:
            for i in range(self.config.max_gleanings):
                # On réutilise le contexte de la conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": CONTINUE_PROMPT})
                
                response = await self.llm.ask(messages)

                full_results += f"\n---\n{response}" 

                # Loop check (LOOP_PROMPT)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": LOOP_PROMPT})
                check = await self.llm.ask(messages)
                if "Y" not in check.upper():
                    break
        
        return full_results


    async def __call__(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        source_id: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Point d'entrée principal.
        Prend un texte + métadonnées et retourne (Entities_DF, Relationships_DF).
        """
        # 1. Préparer le contexte (Fiche d'identité)
        doc_context = self.parser.format_document_context(metadata)
        
        # 2. Lancer l'extraction initiale et la boucle de gleaning
        raw_results = await self._extract_with_gleaning(text, doc_context)
        
        # 3. Parser les résultats bruts en DataFrames
        return self._process_to_dataframes(raw_results, source_id)

    async def _extract_with_gleaning(self, text: str, context: str) -> str:
        """Boucle d'extraction (Gleaning) calquée sur Microsoft."""
        
        # Construction du prompt initial
        prompt_input = GRAPH_EXTRACTION_PROMPT.format(
            entity_types=",".join(self.config.entity_types),
            document_metadata=context,
            input_text=text
        )

        messages = [{"role": "user", "content": prompt_input}]
        
        # Première passe
        response = await self.llm.ask(messages)
        full_results = response
        messages.append({"role": "assistant", "content": response})

        # Boucle de Gleaning (Si max_gleanings > 0)
        if self.config.max_gleanings > 0:
            for i in range(self.config.max_gleanings):
                # On demande de continuer
                messages.append({"role": "user", "content": CONTINUE_PROMPT})
                response = await self.llm.ask(messages)
                
                full_results += f"\n{self.config.record_delimiter}\n{response}"
                messages.append({"role": "assistant", "content": response})

                if i >= self.config.max_gleanings - 1:
                    break

                # Vérification : Reste-t-il quelque chose ?
                messages.append({"role": "user", "content": LOOP_PROMPT})
                check = await self.llm.ask(messages)
                if "Y" not in check.upper():
                    break
        
        return full_results

    def _process_to_dataframes(self, raw_text: str, source_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transforme le texte brut en DataFrames via le Parser."""
        tuples = self.parser.to_tuples(raw_text, self.config.tuple_delimiter)
        
        entities = []
        relationships = []

        for t in tuples:
            if not t: continue
            
            tag = t[0].lower().replace('"', '').replace("'", "")
            
            if tag == "entity" and len(t) >= 4:
                entities.append({
                    "title": t[1].upper(),
                    "type": t[2].upper(),
                    "description": t[3],
                    "source_id": source_id
                })
            elif tag == "relationship" and len(t) >= 5:
                try: strength = float(t[4])
                except: strength = 1.0
                
                relationships.append({
                    "source": t[1].upper(),
                    "target": t[2].upper(),
                    "description": t[3],
                    "weight": strength,
                    "source_id": source_id
                })

        # Création des DataFrames
        ent_df = pd.DataFrame(entities) if entities else pd.DataFrame(columns=["title", "type", "description", "source_id"])
        rel_df = pd.DataFrame(relationships) if relationships else pd.DataFrame(columns=["source", "target", "weight", "description", "source_id"])
        
        return ent_df, rel_df