from app.services.llm.tracker import LLMTracker

def test_tracker_cost_calculation():
    """Vérifie le calcul du coût pour gpt-4o-mini."""
    tracker = LLMTracker()
    # Simuler 1,000,000 tokens d'entrée (0.15$) et 1,000,000 de sortie (0.60$)
    tracker.add_usage(prompt_tokens=1000000, completion_tokens=1000000, model_name="gpt-4o-mini")
    
    # Coût attendu : 0.15 + 0.60 = 0.75$
    assert tracker.usage.total_cost == 0.75
    assert tracker.usage.total_tokens == 2000000

def test_tracker_report_format():
    """Vérifie que le rapport est bien formaté pour l'affichage."""
    tracker = LLMTracker()
    tracker.add_usage(10, 5, "gpt-4o-mini")
    report = tracker.get_report()
    
    assert "Tokens: 15" in report
    assert "Cost: $" in report