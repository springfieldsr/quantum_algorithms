import cirq
import random

import numpy as np

from tqdm import tqdm
from Deutsch_Jozsa import deutsch_jozsa_solver
from Bernstein_Vazirani import bernstein_vazirani_solver
from Simon import simon_solver

def dj_random_test(n, number_of_tests):
    for _ in tqdm(range(number_of_tests)):
        constant_balanced_flag = random.choice([0, 1])
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

        try:
            assert deutsch_jozsa_solver(f, n) == constant_balanced_flag
        except:
            print("DJ Test Failed. Returning the failed test case function...")
            return function_mapping
    
    print("Deutsch_Jozsa Solver all clear for {} bits and {} test.".format(n, number_of_tests))
    return


def bv_random_test(n, number_of_tests):
    for _ in tqdm(range(number_of_tests)):
        a = [random.choice([0, 1]) for _ in range(n)]
        b = random.choice([0, 1])

        def f(binary):
            int_binary = [int(i) for i in binary]
            dot_product = sum([int_binary[-i] * a[-i] for i in range(1, 1 + len(int_binary))]) % 2
            return str((b + dot_product) % 2)
        
        pred_a, pred_b = bernstein_vazirani_solver(f, n)
        try:
            assert "".join([str(i) for i in a]) == pred_a
            assert str(b) == pred_b
        except:
            print("BV Test Failed. Returning the failed test case function...")
            return a, b
    
    print("Bernstein_Vazirani Solver all clear for {} bits and {} tests.".format(n, number_of_tests))
    return


def simon_random_test(n, number_of_tests):
    for _ in tqdm(range(number_of_tests)):
        s = [random.choice([0, 1]) for _ in range(n)]
        while sum(s) == 0:
            s = [random.choice([0, 1]) for _ in range(n)]
        function_mapping = {}
        function_range = random.sample(list(range(2**n)), 2**(n-1))
        for i in range(2**(n-1)):
            binary = "{0:b}".format(function_range[i])
            padding = "0" * (n- len(binary))
            binary = padding + binary
            function_range[i] = binary
        
        function_range_index = 0
        print(s)
        for i in range(2**n):
            if i not in function_mapping.keys():
                print(i)
                function_mapping[i] = function_range[function_range_index]
                function_range_index += 1

                binary = bin(i)[2:]
                int_binary = [int(i) for i in binary]
                counterpart = s[:(n - len(int_binary))] + \
                    [(s[-i] + int_binary[-i]) % 2 for i in range(1, 1 + len(int_binary))][::-1]
                counterpart = "".join([str(i) for i in counterpart])
                function_mapping[int(counterpart, 2)] = function_mapping[i]
        
        def f(binary):
            return function_mapping[int(binary, 2)]

        try:
            res = simon_solver(f, n)
            assert res == "".join([str(i) for i in s])
        except:
            print("Simon Test Failed. Returning the failed test case function...")
            return function_mapping, s
    
    print("Simon Solver all clear for {} bits and {} tests.".format(n, number_of_tests))
    return

#dj_random_test(3,5)
#bv_random_test(3,5)
tmp = simon_random_test(2, 1)
if tmp: print(tmp)