
from typing import Any




class IdentityTracker:
    def __init__(self):
        self._mapping = {}

    def add_mapping(self, old_name: Any, new_name: Any):
        # On force en string et on strip les espaces inutiles
        old_s = str(old_name).strip()
        new_s = str(new_name).strip()

        if not old_s or not new_s or old_s == new_s:
            return
            
        self._mapping[old_s] = new_s
        self._mapping[old_s.upper()] = new_s
        self._mapping[old_s.lower()] = new_s

    def resolve(self) -> dict:
        """Applique la logique transitive (A->B, B->C => A->C)"""
        final_map = {}
        for key in self._mapping:
            path = set([key])
            current = self._mapping[key]
            while current in self._mapping and current not in path:
                path.add(current)
                current = self._mapping[current]
            final_map[key] = current
        return final_map