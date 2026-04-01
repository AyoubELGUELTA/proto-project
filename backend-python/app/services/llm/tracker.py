from dataclasses import dataclass

@dataclass
class TokenUsage:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0

class LLMTracker:
    """Responsable unique du suivi de la consommation."""
    def __init__(self):
        self.usage = TokenUsage()

    def add_usage(self, prompt_tokens: int, completion_tokens: int, model_name: str):
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.total_tokens += (prompt_tokens + completion_tokens)
        self.usage.total_cost += self._calculate_cost(prompt_tokens, completion_tokens, model_name)

    def _calculate_cost(self, prompt_t, completion_t, model):
        # On centralise les prix ici, TODO ajouter plusieurs models par la suite avec leur pricing
        prices = {
            "gpt-4o-mini": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
            "gpt-4o": {"in": 5.00 / 1_000_000, "out": 15.00 / 1_000_000},
        }
        m = prices.get(model, prices["gpt-4o-mini"])
        return (prompt_t * m["in"]) + (completion_t * m["out"])

    def get_report(self):
        return f"Tokens: {self.usage.total_tokens} | Cost: ${self.usage.total_cost:.4f}"