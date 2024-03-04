import matplotlib.pyplot as plt
import numpy as np

def plot_fitness_over_generations_reduced(fitness_history, title="Fitness Over Generations", n=10):
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
    norm = np.sqrt(u ** 2 + v ** 2)
    
    
    plt.plot(x, y, 'o-', mfc='r', zorder=1)
    plt.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid", color='blue')
    
    
    plt.plot(x[0], y[0], 'go', markersize=10, label="Start")  
    plt.plot(x[-2], y[-2], 'rx', markersize=10, label="End")  
    
    
    for i, city in enumerate(tour):
        plt.text(x[i], y[i], city)
    
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.savefig(f"images/{title}.png")
