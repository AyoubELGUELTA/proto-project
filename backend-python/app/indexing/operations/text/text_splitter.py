from typing import List
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.models.domain import TextUnit

import logging
logger = logging.getLogger(__name__)

class TextSplitter:
    """
    Security splitter that recursively breaks down TextUnits exceeding token limits.
    
    While Docling handles structural chunking, some elements (like massive tables) 
    might still be too large for the embedding model's context window (e.g., BGE-M3).
    This class ensures all units are compliant while preserving metadata and 
    structural continuity.
    """
    def __init__(self, max_tokens: int = 1200, overlap: int = 150):
        """
        Initializes the splitter with the BGE-M3 tokenizer.
        
        Args:
            max_tokens: Maximum allowed tokens per unit.
            overlap: Token overlap between sub-chunks to preserve context.
        """
        self.max_tokens = max_tokens
        self.overlap = overlap
        
        logger.debug(f"⚙️ Initializing TextSplitter (Max: {max_tokens} tokens, Overlap: {overlap})")
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3") #TODO have to centralize the configuration of the embedding model IN the config
        
        # Langchain uses characters for size, but we target tokens. 
        # Multiplier of 3 is a safe heuristic for BGE-M3 (chars to tokens ratio).
        self.langchain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens * 3,
            chunk_overlap=overlap * 3,
            separators=["\n\n", "\n", "|", ". ", " ", ""],
            keep_separator=True
        )

    def split_units(self, units: List[TextUnit]) -> List[TextUnit]:
        """
        Analyzes a batch of units and splits those that exceed the token limit.
        
        Args:
            units: List of TextUnits to validate.
            
        Returns:
            List[TextUnit]: A flattened list containing original and sub-split units.
        """
        if not units:
            return []

        final_units = []
        split_count = 0
        
        logger.info(f"📏 Checking token limits for {len(units)} units...")

        for unit in units:
            text = unit.text or ""
            # Detect if this chunk was originally a table or contained one
            original_had_table = (len(unit.tables) > 0 or "|" in text)
            base_title = unit.metadata.get("heading_refined", "Sans titre")

            # Accurate token count
            token_count = len(self.tokenizer.encode(text))

            if token_count <= self.max_tokens:
                final_units.append(unit)
                continue

            # --- SPLIT REQUIRED ---
            split_count += 1
            sub_texts = self.langchain_splitter.split_text(text)
            num_sub_chunks = len(sub_texts)
            
            logger.debug(f"✂️ Splitting unit {unit.id} ({token_count} tokens) into {num_sub_chunks} sub-units.")

            for i, sub_text in enumerate(sub_texts):
                # Deep copy to preserve original metadata but allow targeted updates
                new_unit = unit.model_copy(deep=True) 
                
                # Logical ID mapping: 'chunk_0' -> 'chunk_0_s1', 'chunk_0_s2'...
                if i > 0:
                    new_unit.id = f"{unit.id}_s{i}"
                
                new_unit.text = sub_text
                
                # Check if this specific sub-chunk still contains table structure
                current_has_pipe = "|" in sub_text
                new_unit.tables = [sub_text] if current_has_pipe else []

                # Metadata enrichment for continuity tracking
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

                # Images are usually kept with the first sub-chunk to avoid duplication
                if i > 0:
                    new_unit.metadata["docling_images"] = []

                final_units.append(new_unit)

        if split_count > 0:
            logger.info(f"✅ Token limit safety: {split_count} units were too large and were subdivided.")
            logger.info(f"📊 Final unit count: {len(final_units)} (Net increase: +{len(final_units) - len(units)}).")
        
        return final_units