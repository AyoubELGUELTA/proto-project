import json
import re
import logging
from typing import Any, List, Dict, Tuple
import pandas as pd
from app.indexing.operations.text.text_utils import normalize_entity_title
logger = logging.getLogger(__name__)

class LLMParser:
    """
    Utility class for transforming raw LLM string responses into structured Python objects.
    
    It supports two main formats:
    1. Standard JSON (with Markdown fence cleaning).
    2. Knowledge Graph Tuples (the GraphRAG specific format using custom delimiters).
    
    This parser is crucial for maintaining data integrity between the unstructured 
    LLM outputs and the structured Pandas DataFrames used for graph construction.
    """
    def __init__(self, tuple_delimiter: str = "<|>"):
        """
        Initializes the parser with a specific record delimiter.
        
        Args:
            tuple_delimiter: The string sequence used to separate elements within a tuple.
        """
        self.tuple_delimiter = tuple_delimiter

    @staticmethod
    def format_document_context(metadata: Dict[str, Any]) -> str:
        """
        Serializes document metadata into a structured text block for prompt injection.
        
        This 'Document Identity Card' guides the LLM during extraction by providing 
        high-level context (Chronology, Core Entities, Summary).
        
        Args:
            metadata: A dictionary containing document-specific attributes.
            
        Returns:
            A formatted string ready to be used in a System or User prompt.
        """
        if not metadata:
            return "No specific document context provided."

        context_parts = []
        
        # 1. High-level identification
        title = metadata.get("TITLE", "Unknown Document")
        subject = metadata.get("SUBJECT_MATTER", "N/A")
        doc_type = metadata.get("DOCUMENT_TYPE", "General")
        
        context_parts.append(f"DOCUMENT TITLE: {title}")
        context_parts.append(f"CATEGORY/TYPE: {doc_type}")
        context_parts.append(f"MAIN SUBJECT: {subject}")

        # 2. Domain-specific markers (Crucial for Sira/Historical texts)
        lang = metadata.get("LANGUAGE", "French")
        chrono = metadata.get("CHRONOLOGY", "N/A")
        context_parts.append(f"CONTEXTUAL LANGUAGE: {lang} | PERIOD: {chrono}")

        # 3. Extraction Guide (Core Entities)
        core_entities = metadata.get("CORE_ENTITIES", [])
        if core_entities:
            context_parts.append(f"CORE ENTITIES TO MONITOR: {', '.join(core_entities)}")

        # 4. Semantic 'Tone' (Executive Summary)
        summary = metadata.get("EXECUTIVE_SUMMARY", "")
        if summary:
            context_parts.append(f"EXECUTIVE SUMMARY: {summary}")

        # 5. Logical Structure
        outline = metadata.get("OUTLINE", [])
        if outline:
            context_parts.append(f"DOCUMENT STRUCTURE: {' > '.join(outline[:5])}...")

        return "\n".join(context_parts)

    @staticmethod
    def to_json(text: str) -> Dict[str, Any]:
        """
        Parses a JSON string from an LLM response, handling Markdown formatting.
        
        Logic:
        - Removes Markdown code blocks (```json ... ```).
        - Uses Regex to isolate the JSON object if the LLM added conversational filler.
        
        Returns:
        - The parsed dictionary or an empty dict if parsing fails.
        """
        try:
            # Strip Markdown code fences
            clean_text = re.sub(r"```json\n?|\n?```", "", text).strip()
            
            # Robust fallback: find the first '{' and last '}'
            json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if json_match:
                clean_text = json_match.group(0)
                
            return json.loads(clean_text)
        except Exception as e:
            print(f"❌ JSON Parsing Failure: {e}")
            return {}
    
    
    @staticmethod
    def to_tuples(text: str, delimiter: str = "<|>") -> List[List[str]]:
        """
        Parses the custom 'GraphRAG' tuple format into a matrix of strings.
        
        Expected Format: 
        (type<|>part1<|>part2) ## (type<|>partA<|>partB)
        
        Args:
            text: The raw string containing segments separated by '##'.
            delimiter: The internal separator within each tuple.
            
        Returns:
            A list of lists, where each inner list contains the cleaned segments.
        """
        results = []
        # Cleanup of specific control tokens
        clean_text = text.replace("<|COMPLETE|>", "").strip()
        
        # Split by knowledge block delimiter
        blocks = clean_text.split("##")
        
        for block in blocks:
            block = block.strip()
            # Ensure the block follows the (content) pattern
            if not block or not block.startswith("("):
                continue
            
            try:
                # Remove surrounding parentheses
                content = block[1:-1] if block.endswith(")") else block[1:]
                
                # Split and clean quotes/whitespace from each segment
                parts = [p.strip().strip('"').strip("'") for p in content.split(delimiter)]
                results.append(parts)
            except Exception as e:
                print(f"⚠️ Malformed block skipped: {block[:50]}... - {e}")
                
        return results


    def to_dataframes(
        self, 
        parsed_results: List[List[List[str]]],
        source_ids: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Converts a collection of parsed tuples into structured DataFrames.
        
        This method processes multiple chunks at once, assigning the correct 
        source_id to each extracted entity or relationship for provenance tracking.
        
        Args:
            parsed_results: A nested list [chunk_index][tuple_index][segment_index].
            source_ids: The list of document/chunk IDs corresponding to each result matrix.
            
        Returns:
            A tuple containing (entities_df, relationships_df).
        """
        all_entities = []
        all_relationships = []
        
        # Iterate over each chunk's result set
        for chunk_tuples, s_id in zip(parsed_results, source_ids):
            for t in chunk_tuples:
                if not t or len(t) < 1: 
                    continue
                
                # Determine type (entity vs relationship)
                tag = t[0].lower()
                
                # Case: ENTITY (Expected: type, title, entity_type, description)
                if tag == "entity" and len(t) >= 4:
                    all_entities.append({
                        "title": normalize_entity_title(t[1]),
                        "type": t[2].upper(),
                        "description": t[3],
                        "source_id": s_id
                    })
                
                # Case: RELATIONSHIP (Expected: type, source, target, description, weight)
                elif tag == "relationship" and len(t) >= 5:
                    try:
                        w = float(t[4])
                    except (ValueError, TypeError):
                        w = 1.0 # Default weight if parsing fails
                        
                    all_relationships.append({
                        "source": normalize_entity_title(t[1]),
                        "target": normalize_entity_title(t[2]),
                        "description": t[3],
                        "weight": w,
                        "source_id": s_id
                    })

        # DataFrame construction with fallback for empty results
        ent_df = pd.DataFrame(all_entities) if all_entities else pd.DataFrame(columns=["title", "type", "description", "source_id"])
        rel_df = pd.DataFrame(all_relationships) if all_relationships else pd.DataFrame(columns=["source", "target", "weight", "description", "source_id"])
        
        return ent_df, rel_df