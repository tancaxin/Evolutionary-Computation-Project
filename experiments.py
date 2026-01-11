import numpy as np
from ga_implementation import genetic_algorithm, FITNESS_THRESHOLD
import time
import csv

"""
COMPLETE Genetic Algorithm Experiments - All Sections
Sections 4a, 4b, 4c, 4d, 4e, 4f

Based on provided fitness function for Student Batch Evolution
"""

print("="*100)
print("COMPLETE GENETIC ALGORITHM EXPERIMENTS - Student Batch Evolution")
print("="*100)
print("\nFitness Function: Student Optimization (7 traits)")
print("  - CGPA, Internship, Attendance, Prof Dev, Peer Eval, Stress Tolerance, Deadline Penalty")
print(f"\nTarget Fitness: {FITNESS_THRESHOLD:.4f}")
print(f"Theoretical Maximum: ~0.85")
print(f"Challenge Level: HIGH (requires {FITNESS_THRESHOLD/0.85*100:.1f}% of theoretical max)")
print("="*100)

# ===== SECTION 4a: MUTATION ONLY =====
experiments_4a = [
    {
        'exp_no': 1,
        'mutation_rate': 0.005,
        'crossover_rate': 0.0,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': False,
        'filename': 'Part1_4a_1.py',
        'description': '4a-1: Mutation only (very low rate 0.005)'
    },
    {
        'exp_no': 2,
        'mutation_rate': 0.01,
        'crossover_rate': 0.0,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': False,
        'filename': 'Part1_4a_2.py',
        'description': '4a-2: Mutation only (low rate 0.01)'
    },
    {
        'exp_no': 3,
        'mutation_rate': 0.02,
        'crossover_rate': 0.0,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': False,
        'filename': 'Part1_4a_3.py',
        'description': '4a-3: Mutation only (medium rate 0.02)'
    },
    {
        'exp_no': 4,
        'mutation_rate': 0.03,
        'crossover_rate': 0.0,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': False,
        'filename': 'Part1_4a_4.py',
        'description': '4a-4: Mutation only (high rate 0.03)'
    },
    {
        'exp_no': 5,
        'mutation_rate': 0.05,
        'crossover_rate': 0.0,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': False,
        'filename': 'Part1_4a_5.py',
        'description': '4a-5: Mutation only (very high rate 0.05)'
    },
]

# ===== SECTION 4b: CROSSOVER ONLY =====
experiments_4b = [
    {
        'exp_no': 6,
        'mutation_rate': 0.0,
        'crossover_rate': 0.7,
        'crossover_type': 'one_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': False,
        'use_crossover': True,
        'filename': 'Part1_4b_1.py',
        'description': '4b-1: Crossover only (one-point)'
    },
    {
        'exp_no': 7,
        'mutation_rate': 0.0,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': False,
        'use_crossover': True,
        'filename': 'Part1_4b_2.py',
        'description': '4b-2: Crossover only (two-point)'
    },
    {
        'exp_no': 8,
        'mutation_rate': 0.0,
        'crossover_rate': 0.7,
        'crossover_type': 'uniform',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': False,
        'use_crossover': True,
        'filename': 'Part1_4b_3.py',
        'description': '4b-3: Crossover only (uniform)'
    },
    {
        'exp_no': 9,
        'mutation_rate': 0.0,
        'crossover_rate': 0.7,
        'crossover_type': 'arithmetic',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': False,
        'use_crossover': True,
        'filename': 'Part1_4b_4.py',
        'description': '4b-4: Crossover only (arithmetic)'
    },
]

# ===== SECTION 4c: MUTATION + CROSSOVER =====
experiments_4c = [
    {
        'exp_no': 10,
        'mutation_rate': 0.005,
        'crossover_rate': 0.7,
        'crossover_type': 'one_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4c_1.py',
        'description': '4c-1: Low mut + One-point cross'
    },
    {
        'exp_no': 11,
        'mutation_rate': 0.005,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4c_2.py',
        'description': '4c-2: Low mut + Two-point cross'
    },
    {
        'exp_no': 12,
        'mutation_rate': 0.005,
        'crossover_rate': 0.7,
        'crossover_type': 'uniform',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4c_3.py',
        'description': '4c-3: Low mut + Uniform cross'
    },
    {
        'exp_no': 13,
        'mutation_rate': 0.010,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4c_4.py',
        'description': '4c-4: Medium mut + Two-point cross'
    },
    {
        'exp_no': 14,
        'mutation_rate': 0.008,
        'crossover_rate': 0.9,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4c_5.py',
        'description': '4c-5: Med mut + High crossover (0.9)'
    },
]

# ===== SECTION 4d: POPULATION SIZE =====
experiments_4d = [
    {
        'exp_no': 15,
        'mutation_rate': 0.008,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 30,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4d_1.py',
        'description': '4d-1: Very small population (30)'
    },
    {
        'exp_no': 16,
        'mutation_rate': 0.008,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 50,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4d_2.py',
        'description': '4d-2: Small population (50)'
    },
    {
        'exp_no': 17,
        'mutation_rate': 0.008,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4d_3.py',
        'description': '4d-3: Standard population (100)'
    },
    {
        'exp_no': 18,
        'mutation_rate': 0.008,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 150,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4d_4.py',
        'description': '4d-4: Large population (150)'
    },
    {
        'exp_no': 19,
        'mutation_rate': 0.008,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 200,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4d_5.py',
        'description': '4d-5: Very large population (200)'
    },
]

# ===== SECTION 4e: CROSSOVER TECHNIQUES (DETAILED) =====
experiments_4e = [
    {
        'exp_no': 20,
        'mutation_rate': 0.01,
        'crossover_rate': 0.7,
        'crossover_type': 'one_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4e_1.py',
        'description': '4e-1: One-point crossover technique'
    },
    {
        'exp_no': 21,
        'mutation_rate': 0.01,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4e_2.py',
        'description': '4e-2: Two-point crossover technique'
    },
    {
        'exp_no': 22,
        'mutation_rate': 0.01,
        'crossover_rate': 0.7,
        'crossover_type': 'uniform',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4e_3.py',
        'description': '4e-3: Uniform crossover technique'
    },
    {
        'exp_no': 23,
        'mutation_rate': 0.01,
        'crossover_rate': 0.7,
        'crossover_type': 'arithmetic',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4e_4.py',
        'description': '4e-4: Arithmetic crossover technique'
    },
]

# ===== SECTION 4f: SELECTION TECHNIQUES =====
experiments_4f = [
    {
        'exp_no': 24,
        'mutation_rate': 0.01,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'tournament',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4f_1.py',
        'description': '4f-1: Tournament selection'
    },
    {
        'exp_no': 25,
        'mutation_rate': 0.01,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'rank',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4f_2.py',
        'description': '4f-2: Rank-based selection'
    },
    {
        'exp_no': 26,
        'mutation_rate': 0.01,
        'crossover_rate': 0.7,
        'crossover_type': 'two_point',
        'pop_size': 100,
        'selection_type': 'roulette',
        'use_mutation': True,
        'use_crossover': True,
        'filename': 'Part1_4f_3.py',
        'description': '4f-3: Roulette wheel selection'
    },
]

# Combine all experiments
all_experiments = {
    '4a': experiments_4a,
    '4b': experiments_4b,
    '4c': experiments_4c,
    '4d': experiments_4d,
    '4e': experiments_4e,
    '4f': experiments_4f,
}


def run_experiments(experiments, section_name, n_runs=5, max_generations=15000):
    """Run experiments and collect results"""
    
    results = []
    
    print(f"\n{'='*100}")
    print(f"SECTION {section_name}")
    print(f"{'='*100}\n")
    
    for exp in experiments:
        print(f"Experiment {exp['exp_no']}: {exp['description']}")
        print(f"  Config: Mut={exp['mutation_rate']}, Cross={exp['crossover_type']}({exp['crossover_rate']}), Pop={exp['pop_size']}, Sel={exp['selection_type']}")
        print(f"  File: {exp['filename']}")
        
        generations_list = []
        fitnesses_list = []
        times_list = []
        
        for run in range(n_runs):
            print(f"    Run {run+1}/{n_runs}...", end=' ', flush=True)
            
            start_time = time.time()
            
            generations, best_ind, best_fitness, history = genetic_algorithm(
                pop_size=exp['pop_size'],
                n_genes=7,
                max_generations=max_generations,
                mutation_rate=exp['mutation_rate'],
                crossover_rate=exp['crossover_rate'],
                selection_method=exp['selection_type'],
                crossover_method=exp['crossover_type'],
                use_mutation=exp['use_mutation'],
                use_crossover=exp['use_crossover'],
                verbose=False
            )
            
            elapsed = time.time() - start_time
            
            generations_list.append(generations)
            fitnesses_list.append(best_fitness)
            times_list.append(elapsed)
            
            status = "✓" if best_fitness >= FITNESS_THRESHOLD else "✗"
            print(f"{status} Gens={generations:5d}, Fit={best_fitness:.5f}, Time={elapsed:5.1f}s")
        
        # Calculate statistics
        avg_gens = np.mean(generations_list)
        std_gens = np.std(generations_list)
        avg_fit = np.mean(fitnesses_list)
        std_fit = np.std(fitnesses_list)
        avg_time = np.mean(times_list)
        success_rate = sum(1 for f in fitnesses_list if f >= FITNESS_THRESHOLD) / n_runs * 100
        
        result = {
            'exp_no': exp['exp_no'],
            'description': exp['description'],
            'mutation_rate': exp['mutation_rate'],
            'crossover_rate': exp['crossover_rate'],
            'crossover_type': exp['crossover_type'],
            'pop_size': exp['pop_size'],
            'selection_type': exp['selection_type'],
            'filename': exp['filename'],
            'avg_generations': avg_gens,
            'std_generations': std_gens,
            'avg_fitness': avg_fit,
            'std_fitness': std_fit,
            'avg_time': avg_time,
            'success_rate': success_rate,
            'use_mutation': exp['use_mutation'],
            'use_crossover': exp['use_crossover']
        }
        
        results.append(result)
        
        print(f"  → Summary: Avg Gens = {avg_gens:.0f} ± {std_gens:.0f}, Avg Fit = {avg_fit:.5f}, Success = {success_rate:.0f}%\n")
    
    return results


def save_results(results, filename):
    """Save results to CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'exp_no', 'description', 'mutation_rate', 'crossover_rate',
            'crossover_type', 'pop_size', 'selection_type', 'filename',
            'avg_generations', 'std_generations', 'avg_fitness', 'std_fitness',
            'avg_time', 'success_rate', 'use_mutation', 'use_crossover'
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ Results saved to {filename}\n")


def print_summary_table(results, section_name):
    """Print formatted results table"""
    print(f"\n{'='*145}")
    print(f"{section_name} - SUMMARY TABLE")
    print(f"{'='*145}")
    print(f"{'No':<4} {'Mutation':<9} {'Crossover':<12} {'Cross%':<7} {'Pop':<5} "
          f"{'Avg Generations':<18} {'Selection':<11} {'Success%':<9} {'Filename':<18}")
    print(f"{'-'*145}")
    
    for r in results:
        print(f"{r['exp_no']:<4} "
              f"{r['mutation_rate']:<9.3f} "
              f"{r['crossover_type']:<12} "
              f"{r['crossover_rate']:<7.2f} "
              f"{r['pop_size']:<5} "
              f"{r['avg_generations']:>8.0f} ± {r['std_generations']:<7.0f} "
              f"{r['selection_type']:<11} "
              f"{r['success_rate']:<9.0f} "
              f"{r['filename']:<18}")
    
    print(f"{'='*145}\n")


# Run all experiments
if __name__ == "__main__":
    all_results = {}
    
    # Run each section
    for section_id, experiments in all_experiments.items():
        section_name = f"{section_id}: {experiments[0]['description'].split(':')[0].split('-')[0]}"
        results = run_experiments(experiments, section_id, n_runs=5)
        all_results[section_id] = results
        
        # Save section results
        filename = f'results_{section_id}.csv'
        save_results(results, filename)
        print_summary_table(results, f"SECTION {section_id}")
    
    # Save combined results
    combined_results = []
    for section_results in all_results.values():
        combined_results.extend(section_results)
    save_results(combined_results, 'results_all_sections.csv')
    
    print("\n" + "="*100)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*100)
    print("\nGenerated files:")
    for section_id in all_experiments.keys():
        print(f"  • results_{section_id}.csv")
    print("  • results_all_sections.csv (combined)")
    print("  • fitness.py (provided fitness function)")
    print("  • ga_implementation.py (GA core)")
    print("="*100)




