"""Factory for search policy instances (no Pygame / UI dependencies)."""

from __future__ import annotations

SEARCH_POLICY_NAMES = [
    "Best-First",
    "Beam Search",
    "Hill Climb",
    "A* Belief",
    "Genetic Alg",
]


def make_search_policy(name: str, seed: int = 42):
    """Create a SearchPolicy instance by display name."""
    if name == "Best-First":
        from algorithms.search_policies.best_first_search import BestFirstSearchPolicy

        return BestFirstSearchPolicy()
    if name == "Beam Search":
        from algorithms.search_policies.beam_search import BeamSearchPolicy

        return BeamSearchPolicy()
    if name == "Hill Climb":
        from algorithms.search_policies.hill_climbing import HillClimbingPolicy

        return HillClimbingPolicy(seed=seed)
    if name == "A* Belief":
        from algorithms.search_policies.astar_belief import AStarBeliefPolicy

        return AStarBeliefPolicy()
    if name == "Genetic Alg":
        from algorithms.search_policies.genetic_algorithm import GeneticAlgorithmPolicy

        return GeneticAlgorithmPolicy(seed=seed)
    from algorithms.search_policies.best_first_search import BestFirstSearchPolicy

    return BestFirstSearchPolicy()
