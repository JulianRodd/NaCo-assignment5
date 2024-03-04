from traveling_salesman_problem import fitness_eval
def two_opt(tour, cities_dict, n_cities):
    improvement = True
    tour = list(tour)  
    while improvement:
        improvement = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                new_tour = tour[:i] + list(reversed(tour[i:j+1])) + tour[j+1:]
                if fitness_eval(new_tour, cities_dict, n_cities) < fitness_eval(tour, cities_dict, n_cities):
                    tour = new_tour
                    improvement = True
    return tour
