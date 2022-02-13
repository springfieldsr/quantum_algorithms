import cirq
import numpy as np

from oracle import Oracle
from cirq import Simulator

def simon_circuit(f, n):
    qubits = cirq.LineQubit.range(n + n)
    oracle = Oracle(f, n, n)

    def s_circuit():
        for i in range(n + n):
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
    y_list = []
    circuit = simon_circuit(f, n)
    simulator = Simulator()

    s = None
    for iter in range(20):
        for i in range(n - 1):
            result = simulator.run(circuit)
            measurements = result.data.values.tolist()[0]
            y_list.append(measurements)
        
        y_list = np.array(y_list)
        zeros = np.zeros(n - 1)
        try:
            s = np.linalg.solve(y_list, zeros)
            if sum(s) != 0:
                return s
        except:
            continue
    
    if not s:
        print("Fail to find s with success prob over 99%, please try the solver again.")
        return -1
    return s