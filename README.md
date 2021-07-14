# pyGeneticAlgorithm
Genetic algorithms (GA) for discrete and continuous functions in python

In this repository I implement genetic algorithms in python and provide a few use cases. Any suggestions for improving the algorithm or optimizating the procedure or ideas for use cases are appreciated. At the moment it is just a side project but I am always looking to improve my work.

# Usage

The user needs to define a fitness function to maximize which has to be positve valued. Next, depending on the nature of the decision variables, discrete or continuous, choose the appropriate solver. If a continuous fitness function can be numbafied then the user has the ability to use the continuous numbafied solver. 

For discrete variables one can choose:

**p**:probability of mutation<br>
**pop**: population size<br>
**universe_type**: "equal" or "specific" depending on whether each variable can take on the same values or each variable can take on different values<br>
**universe**: a 1d array or list of possible values each variable can take if universe is equal or a list of lists if specific. In the latter case, there must be as many lists as variables, each list indicating the values each variable can take<br>
**crossover**: midpoint or crossover<br>
**parent contribution**: which % of attributes to take from one parent<br>
**length**: number of variables 
Then, using the solve method, run the algorithm for n_iters of iterations.

For continuous variables one can choose:

**p**:probability of mutation<br>
**pop**: population size<br>
**ub**: vector of upper bounds for each variable. Size is the number of variables.<br>
**lb**: vector of lower bounds for each variable. Size is the number of variables.<br>

## Example code
Explanation in Use cases

```python
from pygenetic_algorithm.discrete_solver import discreteGeneticSolver
from string import ascii_lowercase

#Lets try to have the GA guess some string
stringToFind = "genetic algorithms rock"
universe = list(ascii_lowercase) + " "

def fitness(S):
    
    return sum([ s1 == s2 for (s1,s2) in zip(stringToFind,S) ])
    
dGS = discreteGeneticSolver(0.01,"midpoint",universe = universe,length = len(stringToFind),fitness = fitness)
ans = dGS.solve(600)
```

```python
from pygenetic_algorithm.discrete_solver import discreteGeneticSolver
from string import ascii_lowercase
import networkx as nx

#Network clustering 

def modularity(node_assingments):
    
    Q = 0
    m = 2*G.order()
    
    for u in G:
        
        for v in G:
            
            if u != v and node_assingments[u] == node_assingments[v]:
                
                Q += G.has_edge(u,v) - G.degree(u)*G.degree(v)/(m)
                
    return (1 + Q/m) # Q/m between -1 and 1
    
universe = [ list(min(i)) for i in range(1,len(G)+1)]  # the first node can belong to cluster 0 only, the second can belong to that cluster or its own and so on

dsG = discreteGeneticSolver(0.01,"midpoint",universe, len(universe),modularity, 10000,universe_type="specific")
ans = dGS.solve(600)
```

```python
from pygenetic_algorithm.continuous_solver import continuousGeneticSolver
#Lets try to have the GA the parameters of a system of ordinary differential equations
#here k1 and k2

from scipy.integrate import odeint
import numpy as np

def diffeq(ABC,t,k1,k2):
    
    A,B,C = ABC
    dA = -k1*A**2
    dB = k1*A**2 - k2*B
    dC = k2*B
    return [dA,dB,dC]

ABC0 = [100,0,0]
t = np.linspace(0,250,101)

true_k1 = 0.01
true_k2 = 0.1
observed = odeint(diffeq, ABC0, t, args=(true_k1, true_k2))
  
def fitness(K):
    k1,k2 = K
    ABC0 = [100,0,0]
    sol = odeint(diffeq, ABC0, t, args=(k1, k2))
    return (1/np.linalg.norm(sol - observed))
    
    
cgs = continuousGeneticSolver(0.01,fitness,[0,0],[10,10],1000)   # bounds of each param from 0 to 10
    
```

# Discrete GA

Genetic algorithms randomly search solutions to a given optimization problem drawing inspiration from biology. They try to imitate natural selection in order to evolve a population of solutions to search for better solutions to the problem. Given enough time (though for large problems you probably will not have enough) they can find the global optimum of a function.

The first thing you will need is to define a function to optimize, the so called fitness function. This function takes in an array of values, each position representing a decision variable that can take on a discrete set of values, and returns a positive number.

Imagine you want an algorithm to guess a string, such as "genetic algorithms rock". A pretty contrived example, taken from XXX, but I think is very intuitive. Don't worry, later we will use GA for more useful problems such as feature selection in machine learning and community detection in networks. 

Anyways, imagine that the algorithm knows the length of the string (23 characters), and all it has to do is search for which characters to put in each position. We must then define a function which takes as an argument a string (or array of characters) and judges how close this string is to our target string (and therefore judge the *fitness* of it) using the hamming similarity.

```python
stringToFind = "genetic algorithms rock"

def fitness(S):
    
    return sum([ s1 == s2 for (s1,s2) in zip(stringToFind,S) ])
```

What this function does is count the number of characters that two strings, S and stringToFind, have in common. The max value it can take is the length of the strings. 


We define a candidate solution (here called _S_) for any fitness function as an individual, which is nothing more than an n-dimensional array, where n is the number of decision  variables. Here n is 23, with n[0] being the value of the first letter and so on. **not very clear**

In the case of discrete GA such as this one, decision variables can take on any number of discrete possible values. In principle any variable could have a different set of possible values. In this example each variable will be able to take on the same values: all ascii lowercase letters and a space. We define the universe as the values each decision variable can take.

```python
from string import ascii_lowercase
universe = list(ascii_lowercase) + " "
```

Our genetic algorithm will now randomly initialize a number of individuals, say, a 100. We will evaluate each individual's fitness, which could be done in parallel. They will probably be very far off from the target string, but due to chance, some will be better than others, as judged by out fitness score.

```python
import numpy as np
n_pop = 100
length = 23
pop = [np.random.choice(universe, length) for _ in range(n_pop)]
pop_fitness = np.array([fitness(x) for x in pop])
```

Next step comes the selection. Here, we choose which of our individuals are picked to "reproduce" for the next generation. The probability of being picked is proportional to their fitness (which is why it is important for it to be a positive value). In our example, a string with 10 characters in common with our target is 5 times more likely to be picked than a string with 2 characters in common. Reproduction occurs in pairs, that is, we must choose 2 individuals to reproduce in order to generate a new individual (that is, a new solution to our function) through a process called crossover:

```python
prob = fitness/np.sum(pop_fitness)
p1,p2 = np.random.choice(a = len(pop), p = prob), np.random.choice(a = len(self.pop), p = prob)
crossover(pop[p1],pop[p2])
```

The crossover function governs how a new individual (candidate solution) is generated. There are two methods:

Midpoint:

The first half of the new invidual contains the values of one parent, the next half of the other
Instead of 50/50 we could establish any proportion we want:
    
```python    
child = [0 for _ in range(23)]
proportion = 0.5
c = int(np.ceil(0.5*23))
child[:c] = p1[:c]
child[c:] = p2[c:]
crossover(self.pop[p1],self.pop[p2])
```
    
Crossover:

Here, each variable has a probability _proportion_ to come from one parent or the other
```python    
proportion = 0.5
child = [0 for _ in range(23)]
helper_parent = [p1,p2]
for i in range(23):

    child[i] = helper_parent[np.random.choice(a = 2, p = [proportion, 1 - proportion])][i] 
 ```
 
This process is repeated as many times as individuals there are in a population, giving way to a new generation of indivuals. By preferentially combining individuals with high fitness, hopefully this new generation will have better solutions for our problem. For example, if one individual had a correct character in the first half of the string and another in the second half of the string, by combining them in the correct way we can obtain an individual with 2 correct characters. We could also obtain an individual with 0 correct characters in the strings. On a population level we expect the fitness to increase however. Besides, the creation of suboptimal individuals helps explore areas of the input domain, which given time can give way to better solutions.

```python
pop = [select(fitness) for _ in range(len(pop))]
```

Here the population is obtained of entirely new individuals. However, other methodologies include the possibility of keeping a certain % of parents.

Next comes the mutations. Notice how with crossover we can only obtain individuals whose values are combinations of the older generations. If the older generations where missing a certain value in a particular variable we could never obtain that value-variable combination this way. Imagine no individuals had a k in the last letter. It would be impossible to get the correct answer! This problem could worsen as generations go by, because maybe higher fitness individuals, who have more probability of getting picked, share certain properties in the solution space. It is therefore neccessary to inject some variability into the mix, to generate new individuals with different, random properties. We do this in the mutation stage.

In mutation, we go through each individual. Each of its variables has a random probability _p_ of being mutated, here consisting of randomly replacing its value with one in its universe:

```python
p = 0.01
for i in range(len(individual)):

    for j in range(len(pop[i])):

        if np.random.random() < p:

            pop[i][j] = np.random.choice(universe)
     
```

We now repeat this process n number of items. At each iteration we compare that iterations best individual to our current one, changing them if necessary. Thatś all there is to it.


# Continuous GA
# Todo: Use cases

# To do list

Allow paralelization of fitness evaluations and crossover operations
Numbafy discrete solver
Allow mixed variables
