import random
import numpy as np

from tqdm import tqdm
from Deutsch_Jozsa import deutsch_jozsa_solver
from Bernstein_Vazirani import bernstein_vazirani_solver
from Simon import simon_solver
from Grover import Grover
from QAOA import QAOA

"""
All random tests below follow a same strategy, which is to randomly 
create a function satisfying corresponding algorithm needs in the 
input x output space and check if the algorithm gives the desired
output.
"""

# Random test for Deutsch-Jozsa algorithm
def dj_random_test(n, number_of_tests):
    n_failed = 0
    for _ in range(number_of_tests):
        constant_balanced_flag = random.choice([0, 1])                                          # Randomly decide constant or balanced flag
        if constant_balanced_flag == 0:
            val = str(random.choice([0, 1]))                                                    # If constant, randomly choose all 0's or all 1's
            function_range = [val for _ in range(2**n)]
        else:
            function_range = ['0' for _ in range(2**(n-1))] + ['1' for _ in range(2**(n-1))]    # If balanced, keep the outputs count balanced
            random.shuffle(function_range)                                                      # and shuffle the order of outputs
        
        function_mapping = {i: function_range[i] for i in range(2**n)}

        def f(binary):                                                                          # Create the target function by dictionary mapping
            integer = int(binary, 2)
            return function_mapping[integer]

        try:                                                                                    # Assert the output
            assert deutsch_jozsa_solver(f, n) == constant_balanced_flag
        except:
            print("DJ Test Failed. Returning the failed test case function...")
            print(function_mapping)
            n_failed += 1
    
    if n_failed == 0:
        print("DJ Solver all clear for {} bits and {} tests.".format(n, number_of_tests))
    else:
        print("For {} bits, DJ solver failed {} times".format(n, n_failed))
    return n_failed


def bv_random_test(n, number_of_tests):
    n_failed = 0
    for _ in range(number_of_tests):
        a = [random.choice([0, 1]) for _ in range(n)]                                           # Randomly pick input a
        b = random.choice([0, 1])                                                               # Randomly pick input b

        def f(binary):                                                                          # Create target function by definition with a and b
            int_binary = [int(i) for i in binary]
            dot_product = sum([int_binary[-i] * a[-i] for i in range(1, 1 + len(int_binary))]) % 2
            return str((b + dot_product) % 2)
        
        pred_a, pred_b = bernstein_vazirani_solver(f, n)
        try:                                                                                    # Assert output
            assert "".join([str(i) for i in a]) == pred_a
            assert str(b) == pred_b
        except:
            print("BV Test Failed. Returning the failed test case function...")
            print(a, b)
            n_failed += 1
    
    if n_failed == 0:
        print("BV Solver all clear for {} bits and {} tests.".format(n, number_of_tests))
    else:
        print("For {} bits, BV solver failed {} times".format(n, n_failed))
    return n_failed


def simon_random_test(n, number_of_tests):
    n_failed = 0
    for _ in range(number_of_tests):
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

        try:
            assert simon_solver(f, n) == "".join([str(i) for i in s])
        except:
            print("Simon Test Failed. Returning the failed test case function...")
            print(function_mapping, s)
            n_failed += 1
    
    if n_failed == 0:
        print("Simon Solver all clear for {} bits and {} tests.".format(n, number_of_tests))
    else:
        print("For {} bits, Simon solver failed {} times".format(n, n_failed))
    return n_failed

def grover_random_test(number_of_qubits,  number_of_tests):

    def function(input):
        return 1 if input == answer else 0

    grover_result = []
    ground_truth = []
    for _ in range(number_of_tests):
        answer = np.random.randint(low=0,high=2**number_of_qubits)                                  # Randomly choose a key
        ground_truth.append(answer)
        grover = Grover(number_of_qubits, function)                                                 # Instantiate a Grover class
        result = grover.grover_solver()                                                             # Output result is string
        grover_result.append(int(result,2))

    cnt = 0
    try:
        assert grover_result == ground_truth                                                        # Hopefully, we can pass every test
    except:
        for i in range(number_of_tests):                                                            # Count how many times we failed
            if grover_result[i] != ground_truth[i]:
                cnt += 1
        print("In {} tests, our algorithm failed {} times.".format(number_of_tests, cnt))

        return cnt/number_of_tests
    print("Our Grover simulator passed {} tests.".format(number_of_tests))
    return cnt


def QAOA_random_test(n_max_literals, number_of_tests):
    n_failed = 0
    for _ in range(number_of_tests):
        literals = list(range(1, n_max_literals + 1))
        n_clauses = random.randint(10, 30)                                  # Randomly choose clause length
        clauses = []
        literal_set = set()
        for _ in range(n_clauses):                                          # Randomly create list of clauses
            sign1, sign2 = random.choice([-1, 1]), random.choice([-1, 1])
            x1 = random.choice(literals) * sign1
            x2 = random.choice(literals) * sign2
            clauses.append((x1, x2))

            literal_set.add(abs(x1))
            literal_set.add(abs(x2))
        n_literals = len(literal_set)                                       # Get number of literals in clauses

        #TODO: find the upper bound of satisfiable clauses
        t = random.randint(5, 15)
        try:
            assert QAOA(n_literals, t, 20, clauses)
        except:
            print("QAOA Test Failed. Returning the failed test case function...")
            print(clauses)
            n_failed += 1
    
    if n_failed == 0:
        print("QAOA Solver all clear for {} tests.".format(number_of_tests))
    else:
        print("QAOA solver failed {} times".format(n_failed))
    return n_failed



def main():
    num_tests = 25

    print("==================================")
    print("Deutsch-Jozsa Testing:")
    for n_bits in tqdm(range(1, 10)):
        dj_random_test(n_bits, num_tests)

    print("==================================")
    print("Bernstein-Vazirani Testing:")
    for n_bits in tqdm(range(1, 10)):
        bv_random_test(n_bits, num_tests)
    
    print("==================================")
    print("Simon Testing:")
    for n_bits in tqdm(range(2, 8)):
        simon_random_test(n_bits, num_tests)

    print("==================================")
    print("Grover Testing:")
    for n_bits in tqdm(range(2, 20)):
        grover_random_test(n_bits, num_tests)


if __name__ == '__main__':
    main()