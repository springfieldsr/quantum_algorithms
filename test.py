import cirq
import random

import numpy as np
from tqdm import tqdm
from Deutsch_Jozsa import deutsch_jozsa_solver
from Bernstein_Vazirani import bernstein_vazirani_solver
from Simon import make_simon_circuit, make_oracle, post_processing
from collections import Counter

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


def simon_random_test(number_of_qubits, runs_per_test, number_of_tests):
    quantum_res = []                                                                # Store the outout secret strings from Simon's algorithm
    ground_truth = []                                                               # Store the real secret strings, which are randomly picked.

    for i in tqdm(range(number_of_tests)):
        data = []                                                                   # Store results for MLE

        s = np.random.randint(low=0, high=1, size=number_of_qubits)                 # Generate random s, including all-zero string.
        ground_truth.append(''.join([str(x) for x in s]))

        for _ in range(runs_per_test):
            flag = False                                                            # Check linear independency of output
            while not flag:
                input_qubits = [cirq.GridQubit(i, 0) for i in range(number_of_qubits)]          # Define input qubits
                output_qubits = [                                                               # Define output qubits
                    cirq.GridQubit(i + number_of_qubits, 0) for i in range(number_of_qubits)
                ]

                oracle = make_oracle(input_qubits, output_qubits, s)                            # Create oracle for s, where s could be all-zero,
                                                                                                # representing a 1-1 function
                circuit = make_simon_circuit(input_qubits, output_qubits, oracle)               # Create Simon's circuit

                simulator = cirq.Simulator()
                results = [                                                                     # Query the function n-1 times
                    simulator.run(circuit).measurements['result'][0] for _ in range(number_of_qubits - 1)
                ]

                flag = post_processing(data, results)                                           # Classical processing

        freqs = Counter(data)                                                                   # Afterwards, find the most likely result
        res, freq = sorted(freqs.items(), key=lambda x:x[1], reverse=True)[0]
        # Ideally, there should be only one tuple if the function is 2-1. But to make room for mistakes, we allow for more than one s.
        # But if the probability of this MLE is not high enough, then the answer must be all-zero string, namely the function might be 1-1.
        if int(freq) < 0.8 * runs_per_test:
            res = ''.join(['0'] * number_of_qubits)
        quantum_res.append(res)


    print("Our Simon's algorithm handled {} qubits problem, and every test ran at most {} times.".format(number_of_qubits, runs_per_test))
    try:
        assert quantum_res == ground_truth                                                      # Compare the list of results and groundtruth.
    except:
        cnt = 0
        for i in range(number_of_tests):
            if quantum_res[i] != ground_truth[i]: cnt += 1
        print("Simon solver solver failed {} tests out of {}.".format(cnt, number_of_tests))
        print("The ratio of failure is {}.".format(cnt/number_of_tests))
        return

    print("It passed {} random tests.".format(number_of_tests))
    return

def main():

    #dj_random_test(3,5)
    #bv_random_test(3,5)

    # For testing simon's algorithm, we can alter the #qubits, how many times we run the simon's algorithm and how many random tests we implement.
    simon_random_test(number_of_qubits=2,runs_per_test=2, number_of_tests=10)

if __name__ == '__main__':
    main()