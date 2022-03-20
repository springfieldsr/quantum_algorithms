import numpy as np
import  cirq
from Shor import Factorization, Order
import os


class Shor_Test:
    def __init__(self, path):
        self.path = path
        self.ground_truth = []
        self.classic_result = []
        self.quantum_result = []
        self.circuits = []

    def pre_process(self, max_n):
        n = []  # #bits to represent N
        for i in range(2, max_n + 1):  # from 2 to max_n bits
            N = 2 ** i - 1  # N is the all-one i bits binary number
            while True:  # Randomly generate a, such that
                a = np.random.randint(2, N)  # 1 < a < N
                if np.gcd(a, N) == 1:  # and gcd(a, N) == 1
                    break
            order = Order(a, N)
            self.circuits.append(order.circuit)
            try:  # For the purpose of automatic test, allow for any max_n. Thus, it may run out of RAM
                print(order.quantum_order_finder())
                print("Quantum order finder found the order of " + str(a) + " modulo " + str(N))
                n.append(i)
            except:
                print("When finding order of a " + str(i) + " bits number, it runs out of RAM!")
                break

    def write_qasm_file(self, number_of_bits, number_of_tests):

        for i in range(2, number_of_bits +1):
            self.pre_process(i)
            for j in range(len(self.circuits)):
                circuit = self.circuits[j]
                print(circuit)
                qasm_str = circuit.to_qasm()
                with open(self.path + '\shor_' + str(i) + 'bits_' + str(j) + 'test.qasm', 'w') as f:
                    f.write(qasm_str)
            self.circuits.clear()
        return


path = 'qasm/shor'
if not os.path.exists(path):
    os.mkdir(path)
tester = Shor_Test(path)
tester.write_qasm_file(2,1)
