import random

import numpy as np
from matplotlib.pyplot import plot as plt


class Item(object):
    def __init__(self, v, w):
        self.value = v
        self.weight = w


with open("dataset") as fp:
    capacity_line = fp.readline()
    CAPACITY = int(capacity_line)

    values_line = fp.readline()
    values = values_line.split(",")
    values = list(map(lambda x: int(x), values))

    weights_line = fp.readline()
    weights = weights_line.split(",")
    weights = list(map(lambda x: int(x), weights))

    ITEMS = [Item(values[i], weights[i]) for i in range(0, len(values))]

POP_SIZE = 40
GEN_MAX = 500

BEST_INDIVIDUAL = None


def fitness(target):
    total_value = 0
    total_weight = 0
    index = 0
    for i in target:
        if i == 1:
            total_value += ITEMS[index].value
            total_weight += ITEMS[index].weight
        index += 1

    if total_weight > CAPACITY:
        return 0
    else:
        return total_value


def spawn_starting_population(amount):
    return [spawn_individual() for x in range(0, amount)]


def spawn_individual():
    return [random.randint(0, 1) for x in range(0, len(ITEMS))]


def mutate(target):
    r = random.randint(0, len(target) - 1)
    if target[r] == 1:
        target[r] = 0
    else:
        target[r] = 1


def evolve_population(population):
    parents_slice = 0.1
    mutation_chance = 0.08

    fitness_values = list(map(lambda x: fitness(x), population))
    inidividual_prob = list(map(lambda x: x / sum(fitness_values), fitness_values))
    indexes = list(range(0, len(population)))

    parent_length = int(parents_slice * len(population))
    old_generation = population[:parent_length]

    new_generation = []
    desired_length = len(population) - len(old_generation)
    while len(new_generation) < desired_length:
        first_individual = population[np.random.choice(a=indexes, p=inidividual_prob)]
        second_individual = population[np.random.choice(a=indexes, p=inidividual_prob)]
        half = int(len(first_individual) / 2)
        offspring = first_individual[:half] + second_individual[half:]
        if mutation_chance > random.random():
            mutate(offspring)
        new_generation.append(offspring)

    old_generation.extend(new_generation)
    return old_generation


def main():
    evolution = []
    generation = 1
    population = spawn_starting_population(POP_SIZE)
    BEST_INDIVIDUAL = population[0]
    for g in range(0, GEN_MAX):
        print("Generation %d with %d" % (generation, len(population)))
        population = sorted(population, key=lambda x: fitness(x), reverse=True)
        for i in population:
            print("%s, fit: %s" % (str(i), fitness(i)))

        if fitness(BEST_INDIVIDUAL) < fitness(population[0]):
            BEST_INDIVIDUAL = population[0]

        evolution.append(BEST_INDIVIDUAL)
        population = evolve_population(population)
        generation += 1

    print("Best individual: %s, fit: %s " % (str(BEST_INDIVIDUAL), fitness(BEST_INDIVIDUAL)))


if __name__ == "__main__":
    main()
