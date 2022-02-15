import cirq
import numpy as np

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
        if len(y_list) != set(tuple(row) for row in y_list):
            continue
        try:
            for i in range(2**n):
                binary = "{0:b}".format(i)
                padding = "0" * (n - len(binary))
                binary = padding + binary
                binary_array = np.array([int(i) for i in binary])
                y_array = np.array(y_list)
                if sum(y_array.dot(binary_array.T) % 2) == 0:
                    return binary
                
        except:
            continue
    
    if not s:
        print("Fail to find s with success prob over 99%, please try the solver again.")
        return -1
    return "0" * n