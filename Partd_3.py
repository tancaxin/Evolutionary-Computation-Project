"""
Experiment 15: Large Population
"""

from ga_implementation import genetic_algorithm
import numpy as np
import time

NUM_RUNS = 5
print("="*80)
print("EXPERIMENT 15: Large Population")
print("="*80)
print(f"Config: Pop=200, Mut=0.5, Cross=one-point, Sel=tournament")
print(f"\nRunning {NUM_RUNS} times...")
print("="*80)

results = {'gens': [], 'fitness': [], 'time': [], 'individuals': []}

for run in range(NUM_RUNS):
    print(f"\nRun {run+1}/{NUM_RUNS}: ", end="", flush=True)
    
    start = time.time()
    gens, best_ind, best_fit, _ = genetic_algorithm(
        pop_size=200, n_genes=7, max_generations=15000,
        mutation_rate=0.5, crossover_rate=0.7,
        selection_method='tournament', crossover_method='one_point',
        use_mutation=True, use_crossover=True, verbose=False
    )
    elapsed = time.time() - start
    
    results['gens'].append(gens)
    results['fitness'].append(best_fit)
    results['time'].append(elapsed)
    results['individuals'].append(best_ind)
    
    status = "✓" if best_fit >= 0.85 else "✗"
    print(f"{status} Gens={gens:5d}, Fit={best_fit:.5f}, Time={elapsed:5.2f}s")

# Statistics
avg_gens = np.mean(results['gens'])
std_gens = np.std(results['gens'])
avg_fit = np.mean(results['fitness'])
success = sum(1 for f in results['fitness'] if f >= 0.85) / NUM_RUNS * 100

print(f"\n{'='*80}")
print("STATISTICS:")
print(f"{'='*80}")
print(f"GENERATIONS: {avg_gens:.1f} ± {std_gens:.1f}")
print(f"  Min: {min(results['gens'])}  Max: {max(results['gens'])}  Range: {max(results['gens'])-min(results['gens'])}")
print(f"FITNESS: {avg_fit:.6f}")
print(f"SUCCESS: {success:.0f}%")
print(f"Variance: {(std_gens/avg_gens*100) if avg_gens > 0 else 0:.1f}%")
print(f"{'='*80}")