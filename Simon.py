import cirq
import numpy as np

import scipy as sp
from utils import Oracle
from cirq import Simulator

def simon_circuit(f, n):
    qubits = cirq.LineQubit.range(n + n)
    oracle = Oracle(f, n, n)

    def s_circuit():
        for i in range(n):
            yield cirq.H(qubits[i])
        yield oracle.on(*qubits)
        for i in range(n):
            yield cirq.H(qubits[i])
        for i in range(n):
            yield cirq.measure(qubits[i], key='q' + str(i))
    
    circuit = cirq.Circuit()
    circuit.append(s_circuit())
    return circuit


def simon_solver(f, n):
    circuit = simon_circuit(f, n)
    simulator = Simulator()

    s = None
    for _ in range(100):
        y_list = []
        
        for i in range(n - 1):
            result = simulator.run(circuit)
            measurements = result.data.values.tolist()[0]
            y_list.append(measurements)
        try:
            sing_values = sp.linalg.svdvals(y_list)
            tolerance = 1e-5
            if sum(sing_values < tolerance) == 0:  # check if measurements are linearly dependent
                null_space = sp.linalg.null_space(y_list).T[0]
                solution = np.around(null_space, 3)  # chop very small values
                minval = abs(min(solution[np.nonzero(solution)], key=abs))
                solution = (solution / minval % 2).astype(int)  # renormalize vector mod 2
                return ''.join([str(x) for x in solution])
        except:
            continue
    
    if not s:
        print("Fail to find s with success prob over 99%, please try the solver again.")
        return -1
    return "0" * n