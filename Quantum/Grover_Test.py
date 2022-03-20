import numpy as np
import  cirq
from Grover import Grover
import os

class Grover_Test:
    def __init__(self, path):
        self.path = path
        self.ground_truth = []
        self.classic_result = []
        self.quantum_result = []
        self.circuits = []

    def pre_process(self, number_of_qubits,  number_of_tests):

        def function(input):
            return 1 if input == answer else 0

        for _ in range(number_of_tests):
            answer = np.random.randint(low=0,high=2**number_of_qubits)                                  # Randomly choose a key
            self.ground_truth.append(answer)
            grover = Grover(number_of_qubits, function)                                                 # Instantiate a Grover class
            result = grover.grover_solver()                                                             # Output result is string
            self.circuits.append(grover.circuit)
            self.classic_result.append(int(result,2))

        cnt = 0
        try:
            assert self.classic_result == self.ground_truth                                             # Hopefully, we can pass every test
        except:
            print(self.classic_result, self.ground_truth)
            for i in range(number_of_tests):                                                            # Count how many times we failed
                if self.classic_result[i] != self.ground_truth[i]:
                    cnt += 1
            print("In {} tests, our algorithm failed {} times.".format(number_of_tests, cnt))
            return cnt
        print("Our Grover simulator passed {} tests.".format(number_of_tests))
        return cnt

    def write_qasm_file(self, number_of_bits, number_of_tests):
        with open(self.path + '\grover_answer.txt', 'w') as g:
            g.write('\n')
            g.close()
        for i in range(2, number_of_bits+1):
            self.pre_process(i, number_of_tests)
            for j in range(len(self.circuits)):
                qasm_str = self.circuits[j].to_qasm()
                with open(self.path + '\grover_' + str(i) + 'bits_' + str(j) + 'test.qasm', 'w') as f:
                    f.write(qasm_str)
                with open(self.path + '\grover_answer.txt', 'a') as g:
                    g.write(str(i) + "bits answer: " + str(self.ground_truth[0])+'\n')

            self.classic_result.clear()
            self.ground_truth.clear()
            self.circuits.clear()
        return

    def post_process(self):
        return
path = 'qasm/grover'
if not os.path.exists(path):
    os.mkdir(path)
tester = Grover_Test(path)
tester.write_qasm_file(10, 1)
