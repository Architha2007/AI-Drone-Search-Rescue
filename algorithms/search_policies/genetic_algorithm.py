"""
Genetic Algorithm policy for uncertain-target drone search.

Principle
---------
A Genetic Algorithm (GA) optimizes a population of candidate search routes over
multiple generations using the principles of biological evolution:
    Selection  — choose high-fitness individuals to reproduce.
    Crossover  — combine two parent routes to create offspring.
    Mutation   — randomly alter a route to introduce diversity.
    Elitism    — preserve the best individuals unchanged.

In this context, each "individual" is an ordered sequence of k candidate cells
(a candidate search route).  The GA optimizes this sequence with respect to a
fitness function that balances:
    - Expected detection benefit (cumulative discounted belief probability).
    - Travel cost (total Manhattan distance along the route).
    - Battery feasibility (routes exceeding battery budget are penalized).

The GA returns the FIRST cell of the best evolved route as the next destination.
(Only one step is taken per call, consistent with the SearchMission step model.)

Algorithm Details
-----------------
Representation: A route is a list of k distinct traversable cells.

Fitness function (documented):
    fitness(route) = sum_{i=0}^{k-1} [ prob(cell_i) * decay^i ]
                   - lambda * total_travel_cost(route)
                   - penalty * battery_exceeded(route)

    where:
        prob(cell_i)         = belief probability of the i-th cell
        decay                = future-reward discount factor
        total_travel_cost    = sum of Manhattan distances between consecutive cells
        lambda               = travel cost penalty weight
        battery_exceeded     = 1 if route requires more than usable battery, 0 otherwise
        penalty              = large constant to eliminate infeasible routes

Selection: Tournament selection (randomly sample T candidates, choose best).
Crossover: Order-based crossover (OX) — preserves relative order while avoiding
           duplicate cells in the route.
Mutation:  Randomly replace one cell in the route with an unexplored candidate.

Battery awareness
-----------------
Routes are penalized if their estimated battery cost exceeds the usable budget.
Infeasible routes are still retained in the population to allow recovery via
crossover, but they are never selected as the final output.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Optional, Tuple

from algorithms.astar import manhattan_distance
from config import (
    BATTERY_MOVE_COST_EMPTY,
    GA_ELITE_SIZE,
    GA_GENERATIONS,
    GA_MUTATION_RATE,
    GA_POPULATION_SIZE,
)
from simulation.search_mission import SearchPolicy

if TYPE_CHECKING:
    from simulation.belief_map import BeliefMap
    from simulation.environment import DisasterEnvironment

# Number of cells in each candidate route (chromosome length)
ROUTE_LENGTH = 6


class GeneticAlgorithmPolicy(SearchPolicy):
    """
    Genetic Algorithm search policy: evolves candidate search routes over
    multiple generations to maximize expected detection benefit minus travel cost.

    Only the first cell of the best evolved route is used per step.
    """

    name: str = "Genetic Algorithm"

    def __init__(
        self,
        population_size: int = GA_POPULATION_SIZE,
        generations: int = GA_GENERATIONS,
        mutation_rate: float = GA_MUTATION_RATE,
        elite_size: int = GA_ELITE_SIZE,
        route_length: int = ROUTE_LENGTH,
        decay: float = 0.85,
        travel_penalty: float = 0.008,
        battery_penalty: float = 100.0,
        tournament_size: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.route_length = route_length
        self.decay = decay
        self.travel_penalty = travel_penalty
        self.battery_penalty = battery_penalty
        self.tournament_size = tournament_size
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def select_next(
        self,
        current_pos: Tuple[int, int],
        belief_map: "BeliefMap",
        env: "DisasterEnvironment",
        battery_remaining: float,
        safety_reserve: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Evolve a population of candidate routes and return the first cell of
        the best fitness route.
        """
        usable_battery = battery_remaining - safety_reserve
        beliefs = belief_map.get_all_beliefs()

        # Build the pool of candidate cells (reachable, traversable, affordable)
        candidate_pool = self._build_pool(beliefs, env, current_pos, usable_battery)
        if not candidate_pool:
            return None
        if len(candidate_pool) == 1:
            return candidate_pool[0]

        # Limit route length to pool size
        rl = min(self.route_length, len(candidate_pool))

        # Initialize population
        population = self._init_population(candidate_pool, rl)

        # Evolve
        for _ in range(self.generations):
            fitness_scores = [
                self._fitness(route, beliefs, current_pos, usable_battery)
                for route in population
            ]
            population = self._next_generation(population, fitness_scores, candidate_pool, rl)

        # Select best individual
        final_fitness = [
            self._fitness(route, beliefs, current_pos, usable_battery)
            for route in population
        ]
        best_idx = max(range(len(population)), key=lambda i: final_fitness[i])
        best_route = population[best_idx]

        # Return first cell of best route (one step at a time)
        return best_route[0] if best_route else None

    # ------------------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------------------

    def _fitness(
        self,
        route: List[Tuple[int, int]],
        beliefs: dict,
        current_pos: Tuple[int, int],
        usable_battery: float,
    ) -> float:
        """
        Compute fitness of a candidate route.

        fitness = discounted belief sum - travel_penalty * dist - battery_penalty * infeasible
        """
        if not route:
            return float("-inf")

        total_prob = 0.0
        total_dist = manhattan_distance(current_pos, route[0])
        prev = current_pos

        for i, cell in enumerate(route):
            prob = beliefs.get(cell, 0.0)
            total_prob += prob * (self.decay ** i)
            if i > 0:
                total_dist += manhattan_distance(route[i - 1], cell)
            prev = cell

        # Feasibility penalty
        battery_cost = total_dist * BATTERY_MOVE_COST_EMPTY
        infeasible = 1.0 if battery_cost > usable_battery else 0.0

        return total_prob - self.travel_penalty * total_dist - self.battery_penalty * infeasible

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _init_population(
        self,
        pool: List[Tuple[int, int]],
        route_length: int,
    ) -> List[List[Tuple[int, int]]]:
        """Create an initial population of random routes sampled from pool."""
        population = []
        for _ in range(self.population_size):
            route = self._rng.sample(pool, min(route_length, len(pool)))
            population.append(route)
        return population

    def _next_generation(
        self,
        population: List[List[Tuple[int, int]]],
        fitness_scores: List[float],
        pool: List[Tuple[int, int]],
        route_length: int,
    ) -> List[List[Tuple[int, int]]]:
        """Produce the next generation via elitism, selection, crossover, mutation."""
        new_population = []

        # Elitism: carry over top elite_size individuals unchanged
        elite_indices = sorted(
            range(len(population)), key=lambda i: fitness_scores[i], reverse=True
        )[:self.elite_size]
        for idx in elite_indices:
            new_population.append(list(population[idx]))

        # Fill rest with crossover + mutation
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select(population, fitness_scores)
            parent2 = self._tournament_select(population, fitness_scores)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child, pool, route_length)
            new_population.append(child)

        return new_population[:self.population_size]

    def _tournament_select(
        self,
        population: List[List[Tuple[int, int]]],
        fitness_scores: List[float],
    ) -> List[Tuple[int, int]]:
        """Tournament selection: randomly sample T individuals; return best."""
        indices = self._rng.sample(range(len(population)), min(self.tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return population[best_idx]

    def _crossover(
        self,
        parent1: List[Tuple[int, int]],
        parent2: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Order-based crossover (OX):
        Copy a random slice of parent1 into the child, then fill remaining
        positions with cells from parent2 in order, skipping duplicates.
        """
        if len(parent1) < 2:
            return list(parent1)

        size = len(parent1)
        start = self._rng.randint(0, size - 1)
        end = self._rng.randint(start + 1, size)

        child: List[Optional[Tuple[int, int]]] = [None] * size
        child[start:end] = parent1[start:end]
        child_set = set(parent1[start:end])

        fill_pos = end % size
        for cell in parent2:
            if cell not in child_set:
                child[fill_pos] = cell
                child_set.add(cell)
                fill_pos = (fill_pos + 1) % size
                if child[fill_pos] is not None and fill_pos == start:
                    break

        # Replace any remaining None slots
        for i in range(size):
            if child[i] is None:
                for cell in parent1:
                    if cell not in child_set:
                        child[i] = cell
                        child_set.add(cell)
                        break

        return [c for c in child if c is not None]

    def _mutate(
        self,
        route: List[Tuple[int, int]],
        pool: List[Tuple[int, int]],
        route_length: int,
    ) -> List[Tuple[int, int]]:
        """Randomly replace one cell in the route if mutation rate triggers."""
        if not route or self._rng.random() > self.mutation_rate:
            return route

        route_set = set(route)
        alternatives = [c for c in pool if c not in route_set]
        if not alternatives:
            return route

        idx = self._rng.randint(0, len(route) - 1)
        new_cell = self._rng.choice(alternatives)
        mutated = list(route)
        mutated[idx] = new_cell
        return mutated

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def _build_pool(
        self,
        beliefs: dict,
        env: "DisasterEnvironment",
        current_pos: Tuple[int, int],
        usable_battery: float,
    ) -> List[Tuple[int, int]]:
        """
        Build the pool of candidate cells: reachable, traversable, affordable,
        sorted by belief probability (highest first).
        """
        scored = []
        for cell, prob in beliefs.items():
            if prob < 1e-8:
                continue
            if not env.is_traversable(*cell) or cell not in env.reachable_cells:
                continue
            if cell == current_pos:
                continue
            dist = manhattan_distance(current_pos, cell)
            if dist * BATTERY_MOVE_COST_EMPTY > usable_battery:
                continue
            scored.append((prob, cell))

        scored.sort(reverse=True)
        # Limit pool to top 40 cells to keep GA tractable
        return [c for _, c in scored[:40]]
