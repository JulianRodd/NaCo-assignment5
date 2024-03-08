import time
import numpy as np
from tqdm import tqdm
from utils import two_opt
from utils import (
    plot_fitness_over_generations_reduced,
    plot_tour_with_arrows_and_markers,
    create_random_population_set,
    get_all_fitnes,
    mate_population,
    mutate_population,
    progenitor_selection,
)

n_cities = 10
n_population = 10  # Amount of routes in the the population set, the more the better (more likely to find the best solution)
mutation_rate = 0.5  # The rate at which the population (routes) mutates
n_generations = 1500  # The amount of generations the population will go through
experiment_repeats = 10  # The amount of times the experiment will be repeated

np.random.seed(42)
coordinates_list = [
    [x, y]
    for x, y in zip(
        np.random.randint(0, 100, n_cities), np.random.randint(0, 100, n_cities)
    )
]  # Randomly generated coordinates for the cities
names_list = np.array(
    [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "AA",
        "AB",
        "AC",
        "AD",
        "AE",
        "AF",
        "AG",
        "AH",
        "AI",
        "AJ",
        "AK",
        "AL",
        "AM",
        "AN",
        "AO",
        "AP",
        "AQ",
        "AR",
        "AS",
        "AT",
        "AU",
        "AV",
        "AW",
        "AX",
        "AY",
        "AZ",
    ]
)
cities_dict = {x: y for x, y in zip(names_list, coordinates_list)}


def run_ex_5_4(use_ma):
    best_solutions = []
    all_fitness_histories = []
    best_fitness_history = []
    best_overall_solution = None

    for experiment in range(experiment_repeats):
        print(
            f"Experiment {experiment + 1}/{experiment_repeats} using {'MA' if use_ma else 'EA'}"
        )
        print("Creating population with random solutions")
        population_set = create_random_population_set(
            list(cities_dict.keys()), n_population
        )
        print(
            "First solution in population set:",
            *(f"{city} ->" for city in population_set[0]),
        )
        best_solution = [-1, np.inf, np.array([])]
        fitness_history = []

        for generation in tqdm(range(n_generations), desc="Generations Progress", leave=False):
            fitnes_list = get_all_fitnes(
                population_set, cities_dict, n_population, n_cities
            )
            fitness_history.append(fitnes_list.min())
            if fitnes_list.min() < best_solution[1]:
                best_solution = [
                    generation,
                    fitnes_list.min(),
                    population_set[np.argmin(fitnes_list)],
                ]

            progenitor_list = progenitor_selection(population_set, fitnes_list)
            new_population_set = mate_population(progenitor_list)
            mutated_pop = mutate_population(new_population_set, mutation_rate, n_cities)

            if use_ma:
                mutated_pop = [
                    two_opt(individual, cities_dict, n_cities)
                    for individual in mutated_pop
                ]
            population_set = mutated_pop

        best_solutions.append(best_solution[1])
        all_fitness_histories.append(fitness_history)

        if best_solution[1] == np.min(best_solutions):
            best_fitness_history = fitness_history
            best_overall_solution = best_solution[2]

        print(f"\nBest solution in experiment {experiment + 1}: {best_solution[1]}")

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

    return np.mean(best_solutions), np.min(best_solutions)


starttime_ea = time.time()
mean_ea, best_ea = run_ex_5_4(use_ma=False)
endtime_ea = time.time()
mean_ma, best_ma = run_ex_5_4(use_ma=True)
endtime_ma = time.time()

EA_time = endtime_ea - starttime_ea
MA_time = endtime_ma - endtime_ea
# print(f"Mean Best Fitness for EA: {mean_ea}, Best Fitness: {best_ea}, time taken: {EA_time}")
print(
    f"Mean Best Fitness MA: {mean_ma}, Best Fitness: {best_ma}, time taken: {MA_time}"
)
