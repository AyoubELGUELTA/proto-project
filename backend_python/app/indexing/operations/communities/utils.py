import re

def resolve_context_references(text: str, id_mapping: dict) -> str:
    """
    Parcourt le texte à la recherche des blocs 'Entities (...)', 'Relationships (...)'
    et 'SubCommunities (...)' pour remplacer les entiers locaux par les identifiants
    globaux réels de la base de données.
    """
    if not text:
        return text

    def _resolve_block(match, mapping_key: str, label: str) -> str:
        """Fonction générique de résolution de blocs d'identifiants."""
        ids_str = match.group(1)
        resolved_ids = []
        has_more = False
        
        for local_id in re.split(r',\s*', ids_str):
            clean = local_id.strip()
            if "+more" in clean or not clean:
                has_more = True
                continue
            try:
                clean_id = int(clean)
                labels_fallback = {"entities": "entity", "relationships": "rel", "sub_communities": "sub_comm"} #TODO we could simplify this piece of code
                fallback_label = labels_fallback.get(mapping_key, mapping_key)
                global_id = id_mapping.get(mapping_key, {}).get(clean_id, f"unknown_{fallback_label}_{clean_id}")
                resolved_ids.append(str(global_id))
            except ValueError:
                resolved_ids.append(clean)
                
        if has_more:
            resolved_ids.append("+more")
            
        return f"{label} ({', '.join(resolved_ids)})"

    # Application chirurgicale par Regex pour chaque type de référence contextuelle
    processed_text = re.sub(r'Entities \((.*?)\)', lambda m: _resolve_block(m, "entities", "Entities"), text)#TODO by changing the plural here to singular entities -> entity...
    processed_text = re.sub(r'Relationships \((.*?)\)', lambda m: _resolve_block(m, "relationships", "Relationships"), processed_text)#same
    processed_text = re.sub(r'SubCommunities \((.*?)\)', lambda m: _resolve_block(m, "sub_communities", "SubCommunities"), processed_text)#same
    
    return processed_text