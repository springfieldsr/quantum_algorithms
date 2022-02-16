import cirq
import numpy as np

import scipy as sp
import sympy 
from utils import Oracle
from cirq import Simulator

def simon_circuit(f, n):
    qubits = cirq.LineQubit.range(n + n)                        # Create n qubits and n helper qubits
    oracle = Oracle(f, n, n)                                    # Create Uf

    def s_circuit():
        for i in range(n):
            yield cirq.H(qubits[i])                             # Apply H gates to first n qubits
        yield oracle.on(*qubits)                                # Add Uf
        for i in range(n):
            yield cirq.H(qubits[i])                             # Apply H gates to first n qubits
        for i in range(n):
            yield cirq.measure(qubits[i], key='q' + str(i))     # Measure first n qubits
    
    circuit = cirq.Circuit(s_circuit())
    return circuit


def simon_solver(f, n):
    circuit = simon_circuit(f, n)
    simulator = Simulator()

    s = None
    for _ in range(20):                                         # As Simon is a probablistic algorithm, we need some iters to ensure high prob of success
        y_list = []                                             # y equations list
        
        for i in range(n - 1):
            result = simulator.run(circuit)
            measurements = result.data.values.tolist()[0]
            y_list.append(measurements)                         # Append n - 1 times measurements to the list
        try:
            M = sympy.Matrix(y_list)
            if len(M.T.nullspace(iszerofunc=lambda x: x % 2 == 0)) == 0:            # Check if measurements are linearly dependent
                null_space = M.nullspace(iszerofunc=lambda x: x % 2 == 0)[0].T      # Obtain the basis of y matrix's nullspace
                solution = abs(null_space)
                if min(solution) != 0: solution /= min(solution)
                solution = abs(np.array(solution)[0])
                solution = solution % 2                                             # Mod 2 to get secret string
                for i in range(len(solution)):
                    solution[i] = 1 if solution[i] != 0 else 0                      # If there is a fraction, force it to be 1 if it is greater than 1
                flag = False                                                        # To eliminate the problem caused by float point division
                for y in y_list:                                                    # We perform a check on the obtained solution to see if its dot product
                    if sum([y[i] * solution[i] for i in range(n)]) % 2 != 0:        # with all y's is 0
                        flag = True
                        break
                if flag: continue
                res = ''.join([str(x) for x in solution])
                if not s: s = res
                elif res != s:                                                      # If current solution does not match previous solution,
                    return "0" * n                                                  # Then we know s can only be 0
        except:
            continue
    if not s:
        print("Fail to find s with success prob over 99%, please try the solver again.")
        return -1
    return s