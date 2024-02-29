"""
Implement a string search genetic algorithm like the one in the lecture, but now with:
– tournament selection with a tunable parameter K,
– your own target string of length L of (approximately) 15 characters,
– an alphabet Σ containing all 26 lowercase letters and all 26 capital letters,
– crossover with probability pc = 1,
– a tunable mutation rate µ,
– a population size N = 200
– a fitness as defined in the lecture
– generational replacement with no elitism. Note: since you are doing crossover between two parents, you
will now generate two new children at once, so to get a new population of N strings you need to repeat this
N/2 times.
1. Using K = 2 and µ = 1/L, run the algorithm 10 times to measure the time tfinish (in generations) needed to
find the target.
"""

import string
import random
import math

random.seed(42)


def string_search_GA(T):
    target_string = "PoffertjesYummie"
    length = len(target_string)

    alphabet = list(string.ascii_letters) 
    
	K = 2
    u = 1 / length 
    pc = 1
    N = 200
    
    
    for x in T:
        # 1. start with some population of candidate solutions
        population = ["".join([alphabet[random.randint(0, len(alphabet)-1)] for j in range(length)]) for i in range(N)]
        
        # 2. Determine "fitness" f of every solution i in the population
        fitnesses = {}
        for parent in population:
            fitnesses[parent] = [1 if target_string[i] == char else 0 for i, char in enumerate(parent)] / length

        # 3. Select the fittest individuals for reproduction (tournament selection)
		
		
        parents = random.sample(population, k=K)
		for par in parents:
			highest_fitness = 0
            fittest_parent = 0
			p_fitness = fitnesses[par]
			if p_fitness > highest_fitness:
				highest_fitness = p_fitness
                fittest_parent = par
        
        fittest_ind = fittest_parent
                
    
        # 4. Add variation to get a new generation of offspring
        ### crossover with probability pc = 1,
        ### a tunable mutation rate µ,
        

    
        # repeat 2-4

def get_fitness():
	
    

def main():
    string_search_GA(10)
    