class TagsStore:
    # On stocke la string formatée prête à être injectée
    _cached_prompt: str = "Aucun tag disponible."

    @classmethod
    def set_tags(cls, tags_list: list):
        # tags_list est une liste de dicts [{label, description}, ...]
        formatted = [f"- {t['label']} : {t['description']}" for t in tags_list]
        cls._cached_prompt = "\n".join(formatted)

    @classmethod
    def get_prompt_context(cls):
        return cls._cached_prompt