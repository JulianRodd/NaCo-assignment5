import concurrent.futures
import numpy as np
import time
from utils import (two_opt, plot_fitness_over_generations_reduced, plot_tour_with_arrows_and_markers,
                   create_random_population_set, get_all_fitnes, mate_population, mutate_population,
                   progenitor_selection)

n_cities = 20
n_population = 30  # Amount of routes in the the population set, the more the better (more likely to find the best solution)
mutation_rate = 0.3  # The rate at which the population (routes) mutates
n_generations = 1500  # The amount of generations the population will go through
experiment_repeats = 10  # The amount of times the experiment will be repeated

np.random.seed(42)
import tsplib95
problem = tsplib95.load('ex5_4/a280.tsp')
print(problem)
coordinates_list = [problem.node_coords[i] for i in range(1, n_cities + 1)]
# with open("ex5_4/file-tsp.txt", "r") as file:
#     for line in file:
#         x, y = map(float, line.strip().split())
#         coordinates_list.append([x, y])
        
names_list = np.array(
    [
       str(i) for i in range(1, n_cities + 1)
    ]
)
cities_dict = {x: y for x, y in zip(names_list, coordinates_list)}



def run_single_experiment(experiment, use_ma):
    np.random.seed(experiment + 42)  # Ensure reproducibility for each experiment
    print(f"Experiment {experiment + 1} using {'MA' if use_ma else 'EA'}")
    population_set = create_random_population_set(list(cities_dict.keys()), n_population)
    best_solution = [-1, np.inf, np.array([])]
    fitness_history = []

    for generation in range(n_generations):
        fitnes_list = get_all_fitnes(population_set, cities_dict, n_population, n_cities)
        fitness_history.append(fitnes_list.min())
        if fitnes_list.min() < best_solution[1]:
            best_solution = [generation, fitnes_list.min(), population_set[np.argmin(fitnes_list)]]

        progenitor_list = progenitor_selection(population_set, fitnes_list)
        new_population_set = mate_population(progenitor_list)
        mutated_pop = mutate_population(new_population_set, mutation_rate, n_cities)

        if use_ma:
            mutated_pop = [two_opt(individual, cities_dict, n_cities) for individual in mutated_pop]
        population_set = mutated_pop

    return best_solution, fitness_history

def run_ex_5_4_parallel(use_ma, max_workers=10):
    best_solutions = []
    all_fitness_histories = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_experiment, experiment, use_ma) for experiment in range(experiment_repeats)]
        for future in concurrent.futures.as_completed(futures):
            best_solution, fitness_history = future.result()
            best_solutions.append(best_solution)
            all_fitness_histories.append(fitness_history)
    
   
    fitness_values = [solution[1] for solution in best_solutions]  

    best_fitness_index = np.argmin(fitness_values)
    best_fitness_history = all_fitness_histories[best_fitness_index]
    best_overall_solution = best_solutions[best_fitness_index][2] 

    # Plot the results
    plot_fitness_over_generations_reduced(
        best_fitness_history,
        title=f"{'MA' if use_ma else 'EA'} Fitness Over Generations",
    )

    if best_overall_solution is not None:
        plot_tour_with_arrows_and_markers(
            cities_dict,
            best_overall_solution,
            title=f"Best Tour of {'MA' if use_ma else 'EA'}",
        )

    return np.mean(fitness_values), fitness_values[best_fitness_index]

if __name__ == "__main__":
  starttime_ea = time.time()
  mean_ea, best_ea = run_ex_5_4_parallel(use_ma=False)
  endtime_ea = time.time()
  mean_ma, best_ma = run_ex_5_4_parallel(use_ma=True)
  endtime_ma = time.time()

  EA_time = endtime_ea - starttime_ea
  MA_time = endtime_ma - endtime_ea
  print(f"Mean Best Fitness for EA: {mean_ea}, Best Fitness: {best_ea}, time taken: {EA_time}")
  print(
      f"Mean Best Fitness MA: {mean_ma}, Best Fitness: {best_ma}, time taken: {MA_time}"
  )
