import numpy as np
import numba as nb

class continuousGeneticSolver_numba:

    def __init__(self,p,fitness, ub,lb,pop = 100):

        """
        Genetic algorithm for optimization involving continuous variables, numbafied
        The ability to use it depends on wether or not you are able to numbafy your
        fitness function.
        
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
        self.pop = np.array([np.random.uniform(lb,ub) for _ in range(pop)])

    def solve(self,n_iters):

        return _solve(self.pop,n_iters,self.fitness, self.lb, self.ub, self.p)

@nb.jit(nopython = True)
def _solve(pop, n_iters, fitness, lb,ub,p,verbose = True):

    for j in range(n_iters):

        pop_fitness = fitness(pop)
        max_i = np.argmax(pop_fitness)

        if j == 0:

            max_fitness = pop_fitness[max_i]
            max_response = pop[max_i]

        elif pop_fitness[max_i] > max_fitness:

            max_fitness = pop_fitness[max_i]
            max_response = pop[max_i]
        
        #print([ individual for _ in range(len(pop)//4 ) for individual in __select(pop,pop_fitness)])
        for i in range(len(pop)//4):
            
            pop[i*4:(i+1)*4] = __select(pop,pop_fitness)
            
        for i in range(len(pop)):

            pop[i] = __mutate(pop[i],lb,ub,p)
    
    pop_fitness = fitness(pop)
    max_i = np.argmax(pop_fitness)
    if pop_fitness[max_i] > max_fitness:

        max_fitness = pop_fitness[max_i]
        max_response = pop[max_i]
        
    return max_response
@nb.jit(nopython=True)
def __select(pop, pop_fitness):

    prob = pop_fitness/np.sum(pop_fitness)
    p1 = rand_choice_nb(pop, prob)
    p2 = rand_choice_nb(pop, prob)
    return __crossover(p1,p2)

@nb.jit(nopython=True)
def __crossover(p1,p2):

    l = len(p1)
    ret = np.zeros((4, len(p1)))
    ret[0] = p1
    ret[1] = p2

    for i in range(l):
        
        m = np.random.uniform(0,1)
        p_1 = p1[i]
        p_2 = p2[i]
        ret[2][i] = m*p_1 + (1-m)*p_2
        ret[3][i] = m*p_2 + (1-m)*p_1

    return ret

@nb.jit(nopython=True)
def __mutate(individual,lb,ub,p):

    l = len(individual)
    for i in range(l):

        m = np.random.uniform(0,1)
        if m < p:
            
            new_val = np.random.uniform(0, 10)
            individual[i] = new_val
    return individual

@nb.jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
