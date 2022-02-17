import cirq
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Deutsch_Jozsa import deutsch_jozsa_solver
from Bernstein_Vazirani import bernstein_vazirani_solver
# from Simon import make_simon_circuit, make_oracle, post_processing
from Simon import simon_solver
from collections import Counter
from Grover import Grover
import time
import random

def dj_random_test_graph(start, end):
    tarray = []
    narray = []
    for n in range(start, end + 1):
        constant_balanced_flag = 0
        if constant_balanced_flag == 0:
            val = str(random.choice([0, 1]))
            function_range = [val for _ in range(2**n)]
        else:
            function_range = ['0' for _ in range(2**(n-1))] + ['1' for _ in range(2**(n-1))]
            random.shuffle(function_range)
        
        function_mapping = {i: function_range[i] for i in range(2**n)}

        def f(binary):
            integer = int(binary, 2)
            return function_mapping[integer]
    
        start_time = time.time()
        result_flag = deutsch_jozsa_solver(f, n)
        interval_time = time.time() - start_time
        tarray.append(interval_time)
        narray.append(n)

    plt.plot(narray, tarray, marker="*", color = "grey")
    plt.yscale('log')
    plt.xlabel('N:Bits')
    plt.ylabel('Time in Log Scale')
    plt.title('DJ Random test graph')
    plt.show()
    return


def dj_uftest(n):
    for constant_balanced_flag in range(2):
        if constant_balanced_flag == 0:
            val = str(random.choice([0, 1]))
            function_range = [val for _ in range(2**n)]
        else:
            function_range = ['0' for _ in range(2**(n-1))] + ['1' for _ in range(2**(n-1))]
            random.shuffle(function_range)
        
        function_mapping = {i: function_range[i] for i in range(2**n)}

        def f(binary):
            integer = int(binary, 2)
            return function_mapping[integer]
    
        start_time = time.time()
        result_flag = deutsch_jozsa_solver(f, n)
        interval_time = time.time() - start_time
        print(interval_time)

    return


def bv_random_test_graph(start, end):
    tarray = []
    narray = []
    for n in range(start, end + 1):
        a = [random.choice([0, 1]) for _ in range(n)]
        b = random.choice([0, 1])

        def f(binary):
            int_binary = [int(i) for i in binary]
            dot_product = sum([int_binary[-i] * a[-i] for i in range(1, 1 + len(int_binary))]) % 2
            return str((b + dot_product) % 2)

        start_time = time.time()
        pred_a, pred_b = bernstein_vazirani_solver(f, n)
        interval_time = time.time() - start_time
        tarray.append(interval_time)
        narray.append(n)

    plt.plot(narray, tarray, marker="*", color = "grey")
    plt.yscale('log')
    plt.xlabel('N:Bits')
    plt.ylabel('Time in Log Scale')
    plt.title('BV Random test graph')
    plt.show()
    return

def bv_uftest(n):
    # a = [random.choice([1]) for _ in range(n)]
    a = [1,0,1,0,1,0,1,0,1,0,1,0]
    for b in range(2):
        def f(binary):
            int_binary = [int(i) for i in binary]
            dot_product = sum([int_binary[-i] * a[-i] for i in range(1, 1 + len(int_binary))]) % 2
            return str((b + dot_product) % 2)

        start_time = time.time()
        pred_a, pred_b = bernstein_vazirani_solver(f, n)
        interval_time = time.time() - start_time
        print(interval_time)
    return

def simon_random_test_graph(start, end):
    n_failed = 0
    tarray = []
    narray = []
    for n in range(start, end + 1):
        s = [random.choice([0, 1]) for _ in range(n)]                       # Randomly pick the secret string s
        function_mapping = {}                                               # Set the function mapping for target f
        if sum(s) == 0:                                                     # If s = 0, which means f is one-to-one                          
            function_range = list(range(2**n))                              # Then make the function range to be all 2**n vectors
            random.shuffle(function_range)
            for i in range(2**n):
                binary = "{0:b}".format(function_range[i])
                padding = "0" * (n - len(binary))
                binary = padding + binary
                function_mapping[i] = binary
        else:                                                               # If s != 0, then function is two-to-one
            function_range = random.sample(list(range(2**n)), 2**(n-1))     # Then make the function range to be randomly chosen 2**(n-1) vectors
            for i in range(2**(n-1)):
                binary = "{0:b}".format(function_range[i])
                padding = "0" * (n - len(binary))
                binary = padding + binary
                function_range[i] = binary
            
            function_range_index = 0
            for i in range(2**n):
                if i not in function_mapping.keys():                                                # If current input x1 is not recorded
                    function_mapping[i] = function_range[function_range_index]                      # Record it in the mapping
                    function_range_index += 1

                    binary = "{0:b}".format(i)
                    padding = "0" * (n - len(binary))
                    binary = padding + binary
                    counterpart = "".join([str((s[j] + int(binary[j])) % 2) for j in range(n)])     # Set x1's counterpart x2 to have the same output
                    function_mapping[int(counterpart, 2)] = function_mapping[i]
        
        def f(binary):                                                                              # Create target function by function mapping
            return function_mapping[int(binary, 2)]
        
        start_time = time.time()
        result = simon_solver(f, n)
        interval_time = time.time() - start_time
        tarray.append(interval_time)
        narray.append(n)

    plt.plot(narray, tarray, marker="*", color = "grey")
    plt.yscale('log')
    plt.xlabel('N:Bits')
    plt.ylabel('Time in Log Scale')
    plt.title('Simon Random test graph')
    plt.show()
    return n_failed

def simon_uftest(n):
    s = [0 for _ in range(n)]                       # Randomly pick the secret string s
    function_mapping = {}                                               # Set the function mapping for target f
    if sum(s) == 0:                                                     # If s = 0, which means f is one-to-one                          
        function_range = list(range(2**n))                              # Then make the function range to be all 2**n vectors
        random.shuffle(function_range)
        for i in range(2**n):
            binary = "{0:b}".format(function_range[i])
            padding = "0" * (n - len(binary))
            binary = padding + binary
            function_mapping[i] = binary
    else:                                                               # If s != 0, then function is two-to-one
        function_range = random.sample(list(range(2**n)), 2**(n-1))     # Then make the function range to be randomly chosen 2**(n-1) vectors
        for i in range(2**(n-1)):
            binary = "{0:b}".format(function_range[i])
            padding = "0" * (n - len(binary))
            binary = padding + binary
            function_range[i] = binary
        
        function_range_index = 0
        for i in range(2**n):
            if i not in function_mapping.keys():                                                # If current input x1 is not recorded
                function_mapping[i] = function_range[function_range_index]                      # Record it in the mapping
                function_range_index += 1

                binary = "{0:b}".format(i)
                padding = "0" * (n - len(binary))
                binary = padding + binary
                counterpart = "".join([str((s[j] + int(binary[j])) % 2) for j in range(n)])     # Set x1's counterpart x2 to have the same output
                function_mapping[int(counterpart, 2)] = function_mapping[i]
    
    def f(binary):                                                                              # Create target function by function mapping
        return function_mapping[int(binary, 2)]
    
    start_time = time.time()
    result = simon_solver(f, n)
    interval_time = time.time() - start_time
    print(interval_time)

    return

def main():
    # graphing with start n and end n, inclusive
    # simon_random_test_graph(2,6)
    simon_uftest(4)


if __name__ == '__main__':
    main()