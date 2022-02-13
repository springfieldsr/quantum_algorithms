import cirq

from oracle import Oracle
from cirq import Simulator

def deutsch_jozsa_circuit(f, n):
    qubits = cirq.LineQubit.range(n + 1)
    oracle = Oracle(f, n, 1)

    def DJ_circuit():
        yield cirq.X(qubits[-1])
        for i in range(n + 1):
            yield cirq.H(qubits[i])
        yield oracle.on(*qubits)
        for i in range(n):
            yield cirq.H(qubits[i])
        for i in range(n):
            yield cirq.measure(qubits[i], key='q' + str(i))
    
    circuit = cirq.Circuit()
    circuit.append(DJ_circuit())
    return circuit


def deutsch_jozsa_solver(f, n):
    circuit = deutsch_jozsa_circuit(f, n)
    simulator = Simulator()
    result = simulator.run(circuit)

    measurements = result.data.values.tolist()[0]
    return 0 if sum(measurements) == 0 else 1