import numpy as np

class discreteGeneticSolver:

    def __init__(self, p,crossover_method, universe,length, fitness, \
                 parent_contribution = 0.5,pop = 100, universe_type = "equal" ):

        """
        Genetic algorithm for discrete optimization
        
        There is an objective function to evaluate (fitness) with decision values that can take
        on a finite discrete set of values
        
        An "individual" here is just one assingment of the decision variables
        His "DNA" is composed of the assingment of the decision variables.
        
        A population of size pop starts randomly guessing the answers. Fitness function evaluates them
        Based on this fitness function, each individual has a probability to mate.
        
        During mating, an offspring is produced with part of the "DNA" from one parent and another part 
        from another parent
        
        p: probability of mutation
        crossover_method: how "offspring" are produced from parents
        for the moment, two choices:
            midpoint: select a point. All decisions variables with index up until that point is inherited
            from one parent, and the others from the other parent
            uniform: each decision has a probability to be inherited from one parent or the other
            
        parent_contribution: mating happens randomly between p1 and p2. It is the % of DNA
        you want from parent 1.
        
        length: how long the "DNA strand" is, ie the number of decision variables
        
        Universe and universe_type: the values each decision variable can take. 
            If universe_type is equal, each decision variable can take the same values. 
            If not equal, universe[i] is the values variable i can take
        """

        self.p = p
        self.crossover_method = crossover_method
        self.universe = universe
        self.fitness = fitness
        self.parent_contribution = parent_contribution
        
        if crossover_method not in ("midpoint", "uniform"):

            raise ValueError(f"crossover_method must be midpoint or uniform, not {crossover_method}")
        
        if universe_type not in ("equal", "specific"):

            raise ValueError(f"universe_type must be equal or specific, not {universe_type}")

        if universe_type == "specific":

            if len(universe) != length:

                raise ValueError(f"len(universe) is {len(universe)}, not length!")

            self.pop = [[None]*len(universe) for _ in range(pop)]
            for i in range(pop):

                for j in range(len(universe)):

                    self.pop[i][j] = np.random.choice(universe[j])

        else:

            self.pop = [np.random.choice(universe, length) for _ in range(pop)]

        self.universe_type = universe_type

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
                
            self.pop = [self.select(fitness) for _ in range(len(self.pop))]

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
        child = [0 for _ in range(l)]
        c = int(np.ceil(l*self.parent_contribution))

        if self.crossover_method == "midpoint":

            child[:c] = p1[:c]
            child[c:] = p2[c:]

        elif self.crossover_method == "uniform":

            helper_parent = [p1,p2]
            for i in range(l):

                child[i] = helper_parent[np.random.choice(a = 2, p = [self.parent_contribution, 1 - self.parent_contribution])][i] 
        return child


    def mutate(self,individual):

        for i in range(len(self.pop[individual])):

            if np.random.random() < self.p:

                if self.universe_type == "equal":

                    self.pop[individual][i] = np.random.choice(self.universe)
                else:
                    self.pop[individual][i] = np.random.choice(self.universe[i])
