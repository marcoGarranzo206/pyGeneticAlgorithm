# pyGeneticAlgorithm
Genetic algorithms (GA) for discrete and continuous functions in python

In this repository I code from scratch genetic algorithms in python and provide a few use cases. Any suggestions for improving the algorithm or optimizating the procedure or ideas for use cases are appreciated. At the moment it is just a side project but I am always looking to improve my work.


# Todo: explain discrete GA

Genetic algorithms randomly search solutions to a given optimization problem drawing inspiration from biology. They try to imitate natural selection in order to evolve a population of solutions to search for better solutions to the problem. Given enough time (though for large problems you probably will not have enough) they can find the global optimum of a function.

The first thing you will need is to define a function to optimize, the so called fitness function.

Imagine you want an algorithm to guess a string, such as "genetic algorithms rock". A pretty contrived example, taken from XXX, but I think is very intuitive. Later we will use GA for more useful problems such as feature selection in machine learning and community detection in networks. 

Anyways, imagine that the algorithm knows the length of the string (23 characters), and all it has to do is search for which characters to put in each position. Given a string S, we can judge how close this string is to our target string (and therefore judge the *fitness* of it) using a hamming similarity.

```python
def hamming_similarity(S1, S2):
    
    return sum([ s1 == s2 for (s1,s2) in zip(S1,S2) ])
```

What this function does is count the number of characters that two strings, S1 and S2, have in common. The max value it can take is the length of both strings.
If we want a GA to guess the string "genetic algorithms rock", we fix S1 to be that string, and let S2 be candidate solution S:

```python
stringToFind = "genetic algorithms rock"

def fitness(S):
    
    return sum([ s1 == s2 for (s1,s2) in zip(stringToFind,S) ])
```
We therefore have a function that takes in an iterable (a string, an array...) and spits out a positive number.
We define a candidate solution for any fitness function as an individual, which is nothing more than an n-dimensional array, where n is the number of decision  variables. Here n is 23, with n[0] being the value of the first letter and so on. In the case of discrete GA such as this one, decision variables can take on any number of discrete possible values. In principle any variable could have a different set of possible values, but here each variable will be able to take the same values: all ascii lowercase letters and a space. We define the universe as the values each decision variable can take

```python
from string import ascii_lowercase
universe = list(ascii_lowercase) + " "
```



# Todo: explain continuous GA
# Todo: Use cases
