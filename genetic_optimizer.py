# optimizer/genetic_optimizer.py
import random
import numpy as np

class Train:
    def __init__(self, id, base_speed, priority, position):
        self.id = id
        self.base_speed = base_speed
        self.priority = priority
        self.position = position
        self.speed = base_speed
        self.distance_to_next = 0
        self.distance_to_prev = 0

class SectionOptimizer:
    """
    Genetic Algorithm to maximise section throughput.
    Adjusts speeds to minimise total travel time while avoiding conflicts.
    """
    def __init__(self, trains, section_length=100, safety_distance=10):
        self.trains = trains
        self.section_length = section_length
        self.safety_distance = safety_distance
        self.population_size = 30
        self.generations = 50
        
    def calculate_fitness(self, speed_adjustments):
        total_time = 0
        penalty = 0
        
        for i, train in enumerate(self.trains):
            adj_speed = train.base_speed * max(0.5, min(1.5, speed_adjustments[i]))
            time_to_clear = self.section_length / adj_speed
            total_time += time_to_clear * (1 / train.priority)  # priority weighting
            
            # conflict with train ahead
            if i < len(self.trains) - 1:
                next_train = self.trains[i+1]
                next_speed = next_train.base_speed * max(0.5, min(1.5, speed_adjustments[i+1]))
                rel_speed = adj_speed - next_speed
                separation = next_train.position - train.position
                if rel_speed > 2 and separation < self.safety_distance * 1.5:
                    penalty += 50 * (rel_speed / separation)  # heavy penalty
        
        # maximise throughput = minimise total_time + penalty
        fitness = - (total_time + penalty)
        return fitness
    
    def optimize(self):
        # initial population
        population = []
        for _ in range(self.population_size):
            chromosome = [random.uniform(0.7, 1.3) for _ in self.trains]
            population.append(chromosome)
        
        for _ in range(self.generations):
            fitness = [self.calculate_fitness(chrom) for chrom in population]
            
            # selection (tournament)
            new_pop = []
            for _ in range(self.population_size):
                i1, i2 = random.sample(range(self.population_size), 2)
                parent = population[i1] if fitness[i1] > fitness[i2] else population[i2]
                new_pop.append(parent.copy())
            
            # crossover
            for i in range(0, self.population_size, 2):
                if random.random() < 0.8:
                    point = random.randint(1, len(self.trains)-1)
                    new_pop[i][point:], new_pop[i+1][point:] = new_pop[i+1][point:].copy(), new_pop[i][point:].copy()
            
            # mutation
            for i in range(self.population_size):
                if random.random() < 0.1:
                    pos = random.randint(0, len(self.trains)-1)
                    new_pop[i][pos] = random.uniform(0.7, 1.3)
            
            population = new_pop
        
        # best solution
        best_idx = np.argmax([self.calculate_fitness(chrom) for chrom in population])
        best_adjust = population[best_idx]
        optimal_speeds = [self.trains[i].base_speed * max(0.7, min(1.3, best_adjust[i])) 
                         for i in range(len(self.trains))]
        return optimal_speeds
