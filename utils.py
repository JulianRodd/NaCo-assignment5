import random
from matplotlib import pyplot as plt
def plot_proximity_hist(proximity_hist, n_iterations = 1500, name = "proximity_hist.png"):
    proximity_hist = proximity_hist[:n_iterations]
    plt.title("Fitness over generations")
    plt.plot(proximity_hist)
    plt.hlines(1, 0, n_iterations, colors='r', linestyles='dashed')
    plt.xlabel("Generation")
    plt.ylabel("Fitness (proximity to goal)")
    plt.savefig(f"images/{name}")
    
    
def proximity_to_goal(x, goal):
    return sum([1 for i in range(len(x)) if x[i] == goal[i]]) / len(x)
  
def step1_randomly_generate_bit_sequence(n_bits: int):
    x = [random.choice([0, 1]) for _ in range(n_bits)]
    return x
  
  
def step2_copy_x_invert_bits_with_prob(x, bit_flip_prob):
    x_copy = x
    x_bits_to_flip = [i for i in range(len(x)) if random.random() < bit_flip_prob]
    x_m = [(1 - x_copy[i] if i in x_bits_to_flip else x_copy[i]) for i in range(len(x_copy))]
    return x_m
  
def step3_x_m_better_than_x_with_goal(x, x_m, goal):
    x_proximity_goal = proximity_to_goal(x, goal)
    x_m_proximity_goal = proximity_to_goal(x_m, goal)
    return x_m_proximity_goal > x_proximity_goal
  