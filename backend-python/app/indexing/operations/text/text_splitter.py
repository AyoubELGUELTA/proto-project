from typing import List
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.models.domain import TextUnit

class TextSplitter:
    def __init__(self, max_tokens: int = 1200, overlap: int = 150):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        
        self.langchain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens * 3,
            chunk_overlap=overlap * 3,
            separators=["\n\n", "\n", "|", ". ", " ", ""],
            keep_separator=True
        )

    def split_units(self, units: List[TextUnit]) -> List[TextUnit]:
        final_units = []
        title_counters = {}

        for unit in units:
            text = unit.text
            original_had_table = (len(unit.tables) > 0 or "|" in text)
            
            base_title = unit.metadata.get("heading_refined", "Sans titre")
            if base_title not in title_counters:
                title_counters[base_title] = 0

            token_count = len(self.tokenizer.encode(text))

            if token_count <= self.max_tokens:
                final_units.append(unit)
                continue

            sub_texts = self.langchain_splitter.split_text(text)
            num_sub_chunks = len(sub_texts)

            for i, sub_text in enumerate(sub_texts):
                # 1. COPIE PROFONDE
                new_unit = unit.model_copy(deep=True) 
                
                # Si l'id était 'chunk_0', les nouveaux seront 'chunk_0_s1', 'chunk_0_s2', etc.
                if i > 0:
                    new_unit.id = f"{unit.id}_s{i}"
                
                new_unit.text = sub_text
                
                current_has_pipe = "|" in sub_text
                new_unit.tables = [sub_text] if current_has_pipe else []

                is_cont = (i > 0)
                is_cut = (i < num_sub_chunks - 1)
                is_table_part = original_had_table and current_has_pipe

                suffix = ""
                if i > 0:
                    suffix = f" (Suite Tableau {i})" if current_has_pipe else f" (Suite {i})"
                
                new_unit.metadata.update({
                    "heading_refined": f"{base_title}{suffix}",
                    "is_continuation": is_cont,
                    "is_cut": is_cut,
                    "is_table_continuation": is_table_part and is_cont,
                    "is_table_cut": is_table_part and is_cut,
                    "parent_chunk_id": unit.id  
                })

                if i > 0:
                    new_unit.metadata["docling_images"] = []

                final_units.append(new_unit)
                title_counters[base_title] += 1

        return final_units