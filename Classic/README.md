
# QAOA
The purpose of this is to solve max2sat problem using QAOA on a quantum simulator supported by cirq, given a clause.
The implementation of QAOA involve only:
* QAOA.py

This file contains both classical and quantum solver, random test generator and test function. Solver operates independant of other files but python lib especially cirq. 
The target function, named QAOA, accepts 5 inputs:
* $n_lit$, number of literals in integer
* $t$, minimum number of clauses to satisfy in integer
* $n_trials$, maximum number of trials before abort in integer
* $clauses$, clause in a tuple array, each tuple consists 2 literals, literals being integer
* $verbose$, default false, should set to true if one wants to see the output of QAOA printed

The function outputs the result of max2sat problem by printing binary string value, when verbose is set to true.

# Shor
The purpose of this is to quantumly factor a given number N, using shor algorithm.
The implementation of QAOA involve only:
* Shor.py

This file contains both classical factorization methods and shor algorithm using cirq circuits, random test generator and test function.

The target function is encapsulated in a class called Factorization. To factorize an input integer N, an example code is:
* factorizer = Factorization(N)
* print(factorizer.integer_factorization())

Here integer_factorization() function will perform condition checking and feed inputs into shor algorithm and print output.

* For a prime number checker, we used the Miller Rabin Algorithm: https://gist.github.com/Ayrx/5884790
* For relative arithmetic quantum gates, especially modulo exponential gate, we used some arithmetic gates implementation: https://github.com/kevinddchen/Cirq-PrimeFactorization/blob/main/arithmetic.py
* For reference of the implementation, the following resource are used: https://arxiv.org/abs/quant-ph/9511018

# Below are the readme for last submission 
# Four Quantum Algorithm Solvers  

This project uses Cirq to simulate the implementation of four well known quantum algorithms:  
* Deutsch–Jozsa
* Bernstein-Vazirani
* Simon's
* Grover

# Files  

The implementation of these algorithms involve the following files:  
* Deutsch_Jozsa.py  
* Bernstein_Vazirani.py  
* Simon.py  
* Grover.py  
* utils.py  
The first four files contain the solver for their respective algorithm. The utils file contains functions that create the oracle ($U_f$) matrix for DJ/BV/Simon. The Grover file has its own oracle construction and is wrapped in Grover class.  

We also have two other files to test our implementation:  
* test.py
* graph.py
In test.py, for each solver we create one random test and in graph.py we measure the execution time of each solver and then plot it. 

## Usage  

### Solvers  

Each solver accepts two inputs:  
* $f$, target function  
* $n$, number of bits in domain of $f$  

Such $f$ takes one argument which is a binary string of length $n$. Note that we do not perform explicit check on input $f$ to see whether it satisfies the assumption of a specific algorithm (e.g. either balanced or constant for DJ). Any $f$ that violates algorithm assumption leads to undefined behavior or errors.  

The output of each solver corresponds to the goal of their respective algorithm.  
* DJ solver outputs an int of either 1 or 0, indicating $f$ being balanced or constant, respectively.  
* BV solver outputs two binary strings, $a$ and $b$, of length $n$ and $1$, respectively, indicating the $a$ and $b$ parameters in the definition of BV problem.  
* Simon solver outputs a binary string of length $n$ indicating the secret string $s$ in the definition of Simon's problem.
* Grover solver outputs a binary string of length $n$ indicating the only input of $f$ that gives $1$. Note that we assume Grover's problem here only has $1$ input that yields $1$.  

Also note that for Grover, its solver is wrapped in its class (not a plain method as for DJ/BV/Simon)  

### Testing Files  
For all of the random tests in test.py, they accept two arguments:  
* $n$, number of bits in the domain of $f$  
* number_of_tests, specifying how many tests to perform on a specific solver  

The output of these tests is an int indicating how many times a solver fails with print clauses verbosely reporting the final results.  

The gist of these random test is that we randomly create a mapping from the domain space to range space and use such mapping as $f$ for solvers.  

For graph.py, we simple add graphing functions over the each tests with single runs for us evaluate and analyze the runtime behavior. It is repetitive codes with test.py but with small modifications so it serves as a test ground for our functions, and thus no detailed comments and should not be included in submission.