import numpy as np
import random
from typing import List, Tuple
import copy
from fitness import fitness_function

"""
Genetic Algorithm Implementation for Student Optimization
Based on provided fitness function with 7 traits
"""

# Gene bounds from fitness function
BOUNDS = {
    0: (0.0, 4.0),      # CGPA
    1: (0, 12),         # Internship months
    2: (0, 100),        # Attendance %
    3: (0, 20),         # Professional development certificates
    4: (0, 10),         # Peer evaluation score
    5: (0, 16),         # Stress tolerance (hours/day)
    6: (0, 100),        # Deadline penalty
}

# Maximum fitness (all traits at max, penalty at 0)
# 0.25 + 0.15 + 0.15 + 0.10 + 0.10 + 0.10 - 0 = 0.85
FITNESS_THRESHOLD = 0.85 


def create_individual(n_genes: int = 7) -> List[float]:
    """Create a random individual within bounds"""
    individual = []
    for i in range(n_genes):
        min_val, max_val = BOUNDS[i]
        individual.append(random.uniform(min_val, max_val))
    return individual


def create_population(pop_size: int, n_genes: int = 7) -> List[List[float]]:
    """Create initial population"""
    return [create_individual(n_genes) for _ in range(pop_size)]


# ===== SELECTION METHODS =====

def tournament_selection(population: List[List[float]], 
                        fitnesses: List[float], 
                        tournament_size: int = 3) -> List[float]:
    """Tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
    return copy.deepcopy(population[winner_index])


def rank_selection(population: List[List[float]], 
                   fitnesses: List[float]) -> List[float]:
    """Rank-based selection"""
    sorted_indices = np.argsort(fitnesses)
    ranks = np.arange(1, len(population) + 1)
    probabilities = ranks / ranks.sum()
    chosen_index = np.random.choice(sorted_indices, p=probabilities)
    return copy.deepcopy(population[chosen_index])


def roulette_wheel_selection(population: List[List[float]], 
                             fitnesses: List[float]) -> List[float]:
    """Roulette wheel selection"""
    min_fitness = min(fitnesses)
    adjusted_fitnesses = [f - min_fitness + 0.001 for f in fitnesses]
    total_fitness = sum(adjusted_fitnesses)
    
    if total_fitness == 0:
        return copy.deepcopy(random.choice(population))
    
    probabilities = [f / total_fitness for f in adjusted_fitnesses]
    chosen_index = np.random.choice(len(population), p=probabilities)
    return copy.deepcopy(population[chosen_index])


# ===== CROSSOVER METHODS =====

def one_point_crossover(parent1: List[float], 
                       parent2: List[float]) -> Tuple[List[float], List[float]]:
    """One-point crossover"""
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def two_point_crossover(parent1: List[float], 
                       parent2: List[float]) -> Tuple[List[float], List[float]]:
    """Two-point crossover"""
    points = sorted(random.sample(range(1, len(parent1)), 2))
    child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
    child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
    return child1, child2


def uniform_crossover(parent1: List[float], 
                     parent2: List[float]) -> Tuple[List[float], List[float]]:
    """Uniform crossover"""
    child1 = []
    child2 = []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(g1)
            child2.append(g2)
        else:
            child1.append(g2)
            child2.append(g1)
    return child1, child2


def arithmetic_crossover(parent1: List[float],
                        parent2: List[float],
                        alpha: float = 0.5) -> Tuple[List[float], List[float]]:
    """Arithmetic crossover - blend parents"""
    child1 = [alpha * g1 + (1 - alpha) * g2 for g1, g2 in zip(parent1, parent2)]
    child2 = [(1 - alpha) * g1 + alpha * g2 for g1, g2 in zip(parent1, parent2)]
    return child1, child2


# ===== MUTATION =====

def mutate(individual: List[float], 
          mutation_rate: float) -> List[float]:
    """
    Gaussian mutation respecting gene bounds
    """
    mutated = []
    for i, gene in enumerate(individual):
        if random.random() < mutation_rate:
            min_val, max_val = BOUNDS[i]
            gene_range = max_val - min_val
            mutation_strength = gene_range * 0.01  # 1% of range - very fine tuning
            
            new_gene = gene + random.gauss(0, mutation_strength)
            new_gene = max(min_val, min(max_val, new_gene))
            mutated.append(new_gene)
        else:
            mutated.append(gene)
    return mutated


# ===== MAIN GA =====

def genetic_algorithm(pop_size: int,
                     n_genes: int,
                     max_generations: int,
                     mutation_rate: float,
                     crossover_rate: float,
                     selection_method: str,
                     crossover_method: str,
                     use_mutation: bool = True,
                     use_crossover: bool = True,
                     verbose: bool = False) -> Tuple[int, List[float], float, List[float]]:
    """
    Main genetic algorithm
    Returns: (generations, best_individual, best_fitness, fitness_history)
    """
    # Selection methods
    selection_funcs = {
        'tournament': tournament_selection,
        'rank': rank_selection,
        'roulette': roulette_wheel_selection
    }
    
    # Crossover methods
    crossover_funcs = {
        'one_point': one_point_crossover,
        'two_point': two_point_crossover,
        'uniform': uniform_crossover,
        'arithmetic': arithmetic_crossover
    }
    
    select_func = selection_funcs[selection_method]
    crossover_func = crossover_funcs[crossover_method]
    
    # Initialize population
    population = create_population(pop_size, n_genes)
    fitness_history = []
    
    for generation in range(max_generations):
        # Evaluate fitness
        fitnesses = [fitness_function(ind) for ind in population]
        
        # Track best fitness
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        fitness_history.append(best_fitness)
        
        # Verbose output
        if verbose and generation % 100 == 0:
            print(f"Gen {generation}: Best={best_fitness:.6f}, Avg={avg_fitness:.6f}")
        
        # Check for solution
        if best_fitness >= FITNESS_THRESHOLD:
            best_idx = fitnesses.index(best_fitness)
            if verbose:
                print(f"✓ Solution found at generation {generation + 1}!")
            return generation + 1, population[best_idx], best_fitness, fitness_history
        
        # Create new population
        new_population = []
        
        # Elitism: keep best 2 individuals
        sorted_indices = np.argsort(fitnesses)[::-1]
        new_population.append(copy.deepcopy(population[sorted_indices[0]]))
        if pop_size > 1:
            new_population.append(copy.deepcopy(population[sorted_indices[1]]))
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            parent1 = select_func(population, fitnesses)
            parent2 = select_func(population, fitnesses)
            
            # Crossover
            if use_crossover and random.random() < crossover_rate:
                child1, child2 = crossover_func(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            if use_mutation:
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        population = new_population[:pop_size]
    
    # Return best after max generations
    fitnesses = [fitness_function(ind) for ind in population]
    best_idx = fitnesses.index(max(fitnesses))
    return max_generations, population[best_idx], fitnesses[best_idx], fitness_history


if __name__ == "__main__":
    # Quick test
    print("Testing Student Optimization GA...")
    print(f"Fitness Threshold: {FITNESS_THRESHOLD:.6f}")
    print(f"Theoretical Maximum: ~0.85\n")
    
    generations, best_ind, best_fit, _ = genetic_algorithm(
        pop_size=100,
        n_genes=7,
        max_generations=3000,
        mutation_rate=0.02,
        crossover_rate=0.8,
        selection_method='tournament',
        crossover_method='two_point',
        use_mutation=True,
        use_crossover=True,
        verbose=True
    )
    
    print(f"\nGenerations: {generations}")
    print(f"Best Fitness: {best_fit:.6f}")
    print(f"\nBest Individual:")
    trait_names = ['CGPA', 'Internship', 'Attendance', 'Prof Dev', 
                   'Peer Eval', 'Stress Tol', 'Deadline Penalty']
    for name, value in zip(trait_names, best_ind):
        print(f"  {name:20s}: {value:6.2f}")