import datetime
import numpy as np

# Function to compute the distance between two points
def compute_city_distance_coordinates(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

def compute_city_distance_names(city_a, city_b, cities_dict):
    return compute_city_distance_coordinates(cities_dict[city_a], cities_dict[city_b])


# First step: Create the first population set
def genesis(city_list, n_population):
    population_set = []
    for i in range(n_population):
        # Randomly generating a new solution
        indices = np.random.choice(len(city_list), len(city_list), replace=False)  # Generate indices
        sol_i = [city_list[index] for index in indices]  # Map indices to city names
        population_set.append(sol_i)
    return np.array(population_set)

#2. Evaluation of the fitness

compute_city_distance_names = lambda a,b, cities_dict: np.sqrt((cities_dict[a][0]-cities_dict[b][0])**2 + (cities_dict[a][1]-cities_dict[b][1])**2)
#individual solution
def fitness_eval(city_list, cities_dict, n_cities):
    total = 0
    for i in range(n_cities-1):
        a = city_list[i]
        b = city_list[i+1]
        total += compute_city_distance_names(a,b, cities_dict)
    return total

#All solutions
def get_all_fitnes(population_set, cities_dict, n_population, n_cities):
    fitnes_list = np.zeros(n_population)

    #Looping over all solutions computing the fitness for each solution
    for i in  range(n_population):
        fitnes_list[i] = fitness_eval(population_set[i], cities_dict, n_cities)

    return fitnes_list
  
# 3. Selecting the progenitors
def progenitor_selection(population_set,fitnes_list):
    total_fit = fitnes_list.sum()
    prob_list = fitnes_list/total_fit
    
    #Notice there is the chance that a progenitor. mates with oneself
    progenitor_list_a = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)
    progenitor_list_b = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)
    
    progenitor_list_a = np.array([population_set[i] for i in progenitor_list_a])
    progenitor_list_b = np.array([population_set[i] for i in progenitor_list_b])
    
    
    return np.array([progenitor_list_a,progenitor_list_b])
  

# Pairs crossover
def mate_progenitors(prog_a, prog_b):
    offspring = prog_a[0:5]

    for city in prog_b:

        if not city in offspring:
            offspring = np.concatenate((offspring,[city]))

    return offspring
            
    
# Finding pairs of mates
def mate_population(progenitor_list):
    new_population_set = []
    for i in range(progenitor_list.shape[1]):
        prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
        offspring = mate_progenitors(prog_a, prog_b)
        new_population_set.append(offspring)
        
    return new_population_set

#Offspring production
def mutate_offspring(offspring, mutation_rate, n_cities):
    for q in range(int(n_cities*mutation_rate)):
        a = np.random.randint(0,n_cities)
        b = np.random.randint(0,n_cities)

        offspring[a], offspring[b] = offspring[b], offspring[a]

    return offspring
    
# New populaiton generation
def mutate_population(new_population_set, mutation_rate, n_cities):
    mutated_pop = []
    for offspring in new_population_set:
        mutated_pop.append(mutate_offspring(offspring, mutation_rate, n_cities))
    return mutated_pop
  
