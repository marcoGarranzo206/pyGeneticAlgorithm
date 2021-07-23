import numpy as np

class continuousGeneticSolver:

    def __init__(self,p,fitness, ub,lb,pop = 100):

        """
        Genetic algorithm for optimization involving continuous variables
        
        There is an objective function to evaluate (fitness) with decision values that can take
        on a continuous range of values with a specified range
        
        An "individual" here is just one assingment of the decision variables
        His "DNA" is composed of the assingment of the decision variables.
        
        A population of size pop starts randomly guessing the answers. Fitness function evaluates them
        Based on this fitness function, each individual has a probability to mate.
        
        During mating between two parents, a vector of numbers beta between 0 and 1 is selected for each variable.
        Two children are made from interpolation of the parents variable values:

            for the first, the value of variable i is beta[i]*parent1[i] + (1 - beta[i])*parent2[i]
            for the second, the value of variable i is beta[i]*parent2[i] + (1 - beta[i])*parent1[i]

        Additionally, the two parents remain in the population
        If a decision variable is mutated, a random number is drawn uniformly between its lower and upper bound

        p: probability of mutation
        fitness: fitness function to maximize
        ub:vector of upper bounds. ub[i] is the upper bound for decision variable[i]
        lb:vector of lower bounds. lb[i] is the lower bound for decision variable[i]
        
        pop: population size
        """

        self.p = p
        self.ub = ub
        self.lb = lb
        assert len(ub) == len(lb)
        assert pop % 4 == 0
        self.fitness = fitness
        self.pop = [np.random.uniform(lb,ub) for _ in range(pop)]
        
    def solve(self, n_iters, verbose = True):

        for j in range(n_iters):

            fitness = self.eval_fitness()
            max_i = np.argmax(fitness)

            if j == 0:

                max_fitness = fitness[max_i]
                max_response = self.pop[max_i]

            elif fitness[max_i] > max_fitness:

                max_fitness = fitness[max_i]
                max_response = self.pop[max_i]

            if verbose:

                max_fitness_j = fitness[max_i]
                mean_fitness_j = np.mean(fitness)
                print(f"Max fitness in iteration number {j} : {max_fitness_j}")
                print(f"Mean fitness in iteration number {j} : {mean_fitness_j}")
                
            self.pop = [ individual for _ in range(len(self.pop)//4 ) for individual in self.select(fitness)]
            
            for i in range(len(self.pop)):

                self.mutate(i)
        fitness = self.eval_fitness()
        max_i = np.argmax(fitness)

        if fitness[max_i] > max_fitness:

            max_fitness = fitness[max_i]
            max_response = self.pop[max_i]
        return max_response

    def eval_fitness(self):

        return [self.fitness(x) for x in self.pop]


    def select(self, fitness):

        prob = fitness/np.sum(fitness)
        p1,p2 = np.random.choice(a = len(self.pop), p = prob), np.random.choice(a = len(self.pop), p = prob)

        return self.crossover(self.pop[p1],self.pop[p2])

    def crossover(self,p1,p2):

        l = len(p1)
        child1 = [0 for _ in range(l)]
        child2 = [0 for _ in range(l)]
        mix = np.random.uniform(size= l)
        child1 = [m*p_1 + (1-m)*p_2 for m,p_1,p_2 in zip(mix,p1,p2)]
        child2 = [m*p_2 + (1-m)*p_1 for m,p_1,p_2 in zip(mix,p1,p2)]
        
        return child1,child2,p1,p2


    def mutate(self,individual):

        for i in range(len(self.pop[individual])):

            if np.random.random() < self.p:

                self.pop[individual][i] = np.random.uniform(self.lb[i], self.ub[i])
