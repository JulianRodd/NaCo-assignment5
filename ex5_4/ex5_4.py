import numpy as np
from ex5_4.utils import two_opt
from utils import (
    plot_fitness_over_generations_reduced,
    plot_tour_with_arrows_and_markers,
    create_random_population_set,
    get_all_fitnes,
    mate_population,
    mutate_population,
    progenitor_selection,
)

n_cities = 6
n_population = 100
mutation_rate = 0.3
n_generations = 1500
experiment_repeats = 10

np.random.seed(42)
coordinates_list = [
    [x, y]
    for x, y in zip(
        np.random.randint(0, 100, n_cities), np.random.randint(0, 100, n_cities)
    )
]
names_list = np.array(
    [
        "Amsterdam",
        "Utrecht",
        "Rotterdam",
        "The Hague",
        "Eindhoven",
        "Tilburg",
        "Groningen",
        "Almere",
        "Breda",
        "Nijmegen",
        "Enschede",
        "Apeldoorn",
        "Haarlem",
        "Arnhem",
        "Zaanstad",
        "Amersfoort",
        "s-Hertogenbosch",
        "Haarlemmermeer",
        "Zwolle",
        "Leiden",
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
        print("Creating population set")
        population_set = create_random_population_set(
            list(cities_dict.keys()), n_population
        )
        best_solution = [-1, np.inf, np.array([])]
        fitness_history = []

        for generation in range(n_generations):
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

        print(f"Best solution in experiment {experiment + 1}: {best_solution[1]}")

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


mean_ea, best_ea = run_ex_5_4(use_ma=False)
mean_ma, best_ma = run_ex_5_4(use_ma=True)

print(f"Mean Best Fitness for EA: {mean_ea}, Best Fitness: {best_ea}")
print(f"Mean Best Fitness MA: {mean_ma}, Best Fitness: {best_ma}")
