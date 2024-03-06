import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_over_generations_reduced(
    fitness_history, title="Fitness Over Generations", n=10
):
    plt.figure(figsize=(20, 6))

    reduced_fitness_history = fitness_history[::n]
    plt.plot(reduced_fitness_history, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness of Best Solution")
    plt.grid(True)
    plt.savefig(f"images/{title}.png")


def plot_tour_with_arrows_and_markers(cities_dict, tour, title="Tour"):
    plt.figure(figsize=(10, 6))

    x = [cities_dict[city][0] for city in tour] + [cities_dict[tour[0]][0]]
    y = [cities_dict[city][1] for city in tour] + [cities_dict[tour[0]][1]]

    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u / 2
    pos_y = y[:-1] + v / 2
    norm = np.sqrt(u**2 + v**2)

    plt.plot(x, y, "o-", mfc="r", zorder=1)
    plt.quiver(
        pos_x,
        pos_y,
        u / norm,
        v / norm,
        angles="xy",
        zorder=5,
        pivot="mid",
        color="blue",
    )

    plt.plot(x[0], y[0], "go", markersize=10, label="Start")
    plt.plot(x[-2], y[-2], "rx", markersize=10, label="End")

    for i, city in enumerate(tour):
        plt.text(x[i], y[i], city)

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.savefig(f"images/{title}.png")


def compute_city_distance_coordinates(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def compute_city_distance_names(city_a, city_b, cities_dict):
    return compute_city_distance_coordinates(cities_dict[city_a], cities_dict[city_b])


def create_random_population_set(city_list, n_population):
    population_set = []
    for i in range(n_population):
        # Randomly generating a new solution
        indices = np.random.choice(len(city_list), len(city_list), replace=False)
        cities = [city_list[index] for index in indices]
        population_set.append(cities)
    return np.array(population_set)


def fitness_eval(city_list, cities_dict, n_cities):
    total = 0
    for i in range(n_cities - 1):
        a = city_list[i]
        b = city_list[i + 1]
        total += compute_city_distance_names(a, b, cities_dict)
    return total


def get_all_fitnes(population_set, cities_dict, n_population, n_cities):
    fitnes_list = np.zeros(n_population)

    # Looping over all solutions computing the fitness for each solution
    for i in range(n_population):
        fitnes_list[i] = fitness_eval(population_set[i], cities_dict, n_cities)

    return fitnes_list


def progenitor_selection(population_set, fitnes_list):
    total_fit = fitnes_list.sum()
    prob_list = fitnes_list / total_fit

    # Notice there is the chance that a progenitor mates with oneself
    progenitor_list_a = np.random.choice(
        list(range(len(population_set))), len(population_set), p=prob_list, replace=True
    )
    progenitor_list_b = np.random.choice(
        list(range(len(population_set))), len(population_set), p=prob_list, replace=True
    )

    progenitor_list_a = np.array([population_set[i] for i in progenitor_list_a])
    progenitor_list_b = np.array([population_set[i] for i in progenitor_list_b])

    return np.array([progenitor_list_a, progenitor_list_b])


def mate_progenitors(prog_a, prog_b):
    offspring = prog_a[0:5]

    for city in prog_b:

        if not city in offspring:
            offspring = np.concatenate((offspring, [city]))

    return offspring


def mate_population(progenitor_list):
    new_population_set = []
    for i in range(progenitor_list.shape[1]):
        prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
        offspring = mate_progenitors(prog_a, prog_b)
        new_population_set.append(offspring)

    return new_population_set


def mutate_offspring(offspring, mutation_rate, n_cities):
    for q in range(int(n_cities * mutation_rate)):
        a = np.random.randint(0, n_cities)
        b = np.random.randint(0, n_cities)

        offspring[a], offspring[b] = offspring[b], offspring[a]

    return offspring


def mutate_population(new_population_set, mutation_rate, n_cities):
    mutated_pop = []
    for offspring in new_population_set:
        mutated_pop.append(mutate_offspring(offspring, mutation_rate, n_cities))
    return mutated_pop


def two_opt(tour, cities_dict, n_cities):
    improvement = True
    tour = list(tour)
    while improvement:
        improvement = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                new_tour = tour[:i] + list(reversed(tour[i : j + 1])) + tour[j + 1 :]
                if fitness_eval(new_tour, cities_dict, n_cities) < fitness_eval(
                    tour, cities_dict, n_cities
                ):
                    tour = new_tour
                    improvement = True
    return tour
