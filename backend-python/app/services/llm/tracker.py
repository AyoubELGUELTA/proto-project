from dataclasses import dataclass

@dataclass
class TokenUsage:
    """
    Data structure representing the cumulative token consumption and associated costs.
    """
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0

class LLMTracker:
    """
    Dedicated monitor for LLM consumption and financial overhead.
    
    This class centralizes token counting and cost estimation across all LLM calls.
    It follows the 'Single Responsibility Principle' by decoupling the pricing logic 
    from the service execution logic.
    """
    
    def __init__(self):
        """Initializes the tracker with a fresh TokenUsage counter."""
        self.usage = TokenUsage()

    def add_usage(self, prompt_tokens: int, completion_tokens: int, model_name: str):
        """
        Updates the cumulative usage statistics after a successful LLM call.
        
        Args:
            prompt_tokens: Number of tokens in the input prompt.
            completion_tokens: Number of tokens in the LLM's response.
            model_name: The string identifier of the model used (for pricing lookup).
        """
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.total_tokens += (prompt_tokens + completion_tokens)
        self.usage.total_cost += self._calculate_cost(prompt_tokens, completion_tokens, model_name)

    def _calculate_cost(self, prompt_t: int, completion_t: int, model: str) -> float:
        """
        Calculates the estimated cost based on current provider pricing models.
        
        Prices are normalized to USD per 1,000,000 tokens.
        Note: Pricing is subject to change by providers (OpenAI, Anthropic, etc.).
        """
        # Centralized pricing dictionary (Values as of 2025/2026)
        prices = {
            "gpt-4o-mini": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
            "gpt-4o": {"in": 5.00 / 1_000_000, "out": 15.00 / 1_000_000},
        }
        
        # Fallback to gpt-4o-mini pricing if the model is unknown
        m = prices.get(model, prices["gpt-4o-mini"])
        
        return (prompt_t * m["in"]) + (completion_t * m["out"])

    def get_report(self) -> str:
        """
        Generates a concise summary of the session's consumption.
        
        Returns:
            A formatted string suitable for logging or terminal display.
        """
        return f"Tokens: {self.usage.total_tokens:,} | Cost: ${self.usage.total_cost:.4f}"