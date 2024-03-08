import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from utils import plot_proximity_hist, proximity_to_goal, step1_randomly_generate_bit_sequence, step2_copy_x_invert_bits_with_prob, step3_x_m_better_than_x_with_goal

random.seed(42)

def ex5_2_one_plus_one_genetic_algorithm(n_bits = 10, bit_flip_prob = 0.1, stop_iterations = 1500, compare_fitness = True):
    goal = [1 for _ in range(n_bits)]
    x = step1_randomly_generate_bit_sequence(n_bits)
    initial_proximity = proximity_to_goal(x, goal)
    print(f"Initial x: {x}")
    print(f"Goal: {goal}")
    print(f"Proximity to goal: {proximity_to_goal(x, goal)}")
    loop_count = 0
    flip_steps = 0
    proximity_hist = np.array([])
    with tqdm(total=100, desc="Progress towards goal") as pbar:
        pbar.update(int(initial_proximity * 100))
        fitness = proximity_to_goal(x, goal)
        while fitness < 1 and loop_count < stop_iterations:
            x_m = step2_copy_x_invert_bits_with_prob(x, bit_flip_prob)
            loop_count += 1
            if compare_fitness == False or step3_x_m_better_than_x_with_goal(x, x_m, goal):
                x = x_m
                flip_steps += 1
                pbar.update(int(proximity_to_goal(x, goal) * 100) - pbar.n)
            
            fitness = proximity_to_goal(x, goal)
            proximity_hist = np.append(proximity_hist, fitness)
    
    print(f"Goal reached in {loop_count} loops and {flip_steps} flip steps")
    
    return proximity_hist

# EXERCISE 5.2.1
proximity_hist_5_2_1 = ex5_2_one_plus_one_genetic_algorithm(n_bits = 100, bit_flip_prob = 0.01, stop_iterations = 1500, compare_fitness = True)
plot_proximity_hist(proximity_hist_5_2_1, n_iterations = 1500, name = "ex5_2_1.png")
plt.close()

# EXERCISE 5.2.2
proximity_hist_5_2_2 = ex5_2_one_plus_one_genetic_algorithm(n_bits = 100, bit_flip_prob = 0.01, stop_iterations = 1500, compare_fitness = False)
plot_proximity_hist(proximity_hist_5_2_2, n_iterations = 1500, name = "ex5_2_2.png")
plt.close()

# EXERCISE 5.2.4
plot_proximity_hist(proximity_hist_5_2_1, n_iterations = 1500, name = "ex5_2_4.png")
plot_proximity_hist(proximity_hist_5_2_2, n_iterations = 1500, name = "ex5_2_4.png")