import string
import random
import numpy as np
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

random.seed(42)


def string_search_GA(T, u, length, target_string):
    alphabet = list(string.ascii_letters) 
    
    K = 2
    pc = 1
    N = 200

    target_times = []
    
    for x in range(T):
        hamming_means = []
        hamming_gens = []

        # 1. start with some population of candidate solutions
        population = ["".join([alphabet[random.randint(0, len(alphabet)-1)] for j in range(length)]) for i in range(N)]
        generation = 0

        while target_string not in population:
            generation += 1

            # 2. Determine "fitness" f of every solution i in the population
            fitnesses = {}
            for parent in population:
                fitnesses[parent] = sum([1 if target_string[i] == char else 0 for i, char in enumerate(parent)]) / length
            print(f"Highest fitness in generation {generation} is {max(fitnesses.values())}")

            # 3. Select the fittest individuals for reproduction (tournament selection)
            fit_parents = []
            for i in range(int(N/K)):
                parents = random.sample(population, k=K)
                for parent in parents:
                    population.remove(parent)
                
                if fitnesses[parents[0]] > fitnesses[parents[1]]:
                    fit_parents.append(parents[0])
                else:
                    fit_parents.append(parents[1])
        
            # 4. Add variation to get a new generation of offspring
            ### crossover with probability pc = 1,
            ### a tunable mutation rate µ,

            new_generation = []    
            for i in range(int(N/K/K)):
                parents = random.sample(fit_parents, k=K)
                for parent in parents:
                    fit_parents.remove(parent)
                
                p1 = parents[0] 
                p2 = parents[1]
                
                for i in range(2):
                    # Crossover
                    children = [p1, p2]
                    if random.random() <= pc:
                        crossover_index = random.randint(1, 14)
                        children = [p1[:crossover_index+1] + p2[crossover_index+1:], p2[:crossover_index+1] + p1[crossover_index+1:]]

                    # Mutation 
                    for i in range(len(children)):
                        for char_i in range(len(children[i])):
                            if random.random() <= u:
                                children[i] = children[i][:char_i] + str(alphabet[random.randint(0, len(alphabet)-1)]) + children[i][char_i + 1:] 
                    
                    new_generation += children
                
            population = new_generation

            # Calculate hamming distance for a random selection of 50 samples
            if generation % 10 == 0:
                sample = random.sample(population, k=50)
                hamming_vals = []
                for s in sample:
                    hamming_vals.append(hamming(list(target_string), list(s), w=None))

                mean = np.mean(hamming_vals)
                hamming_means.append(mean)
                hamming_gens.append(generation)
                print(f"Hamming mean: {mean}")
                
            if target_string in population:
                print(f"Target found in generation {generation}")
                target_times.append(generation)
                plt.plot(hamming_gens, hamming_means)
                plt.title("Population diversity over time")
                plt.xlabel("Number of generations")
                plt.ylabel("Average Hamming distance")
                plt.savefig(f"hamm_string_search_T{x}_u{round(u,2)}.png")
                plt.close()
                break

            if generation > 200:
                target_times.append(generation)
                plt.plot(hamming_means)
                plt.title("Population diversity over time")
                plt.xlabel("Number of generations")
                plt.ylabel("Average Hamming distance")
                plt.savefig(f"hamm_string_search_T{x}_u{round(u,2)}.png")
                plt.close()
                break
        
        # repeat 2-4
    
    return (sum(target_times)/len(target_times)) if target_times else None, target_times
            

def main():
    target_string = "PoffertjesYummie"
    length = len(target_string)
    u_list = [0/length, 1/length, 3/length]
    
    for u in u_list:
        avg_num_generations, target_times = string_search_GA(100, u, length, target_string=target_string)
        print(f"It took on average {avg_num_generations} generations to find the target string.")

        data = {
                "target_generation" : target_times 
            }
        generations_df = pd.DataFrame(data)
        plot = sns.swarmplot(data=generations_df)
        plot.axhline(avg_num_generations, xmin=0.30, xmax=0.70, color="red")
        plt.plot([0], [avg_num_generations], color='k', linestyle='-', linewidth=2)
        plt.title("Number of generations before target string is found")
        plt.ylabel("Number of generations")
        plt.savefig(f"seaborn_plot_{round(1/length,2)}.png")


if __name__ == "__main__":
    main()