import cirq
import numpy as np
from cirq import Simulator


class PhaseShift(cirq.Gate):
    def __init__(self, gamma):
        super(PhaseShift, self)
        self.gamma = gamma
    
    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        mat = np.array([[1, 0], [0, np.exp(1j * self.gamma)]])
        return mat

    def _circuit_diagram_info_(self):
        return "P"


def qaoa_circuit(n, gamma, beta, clauses):
    qubits = cirq.LineQubit.range(n + 1)
    P1 = PhaseShift(gamma)
    P2 = PhaseShift(-1 * gamma)

    yield cirq.X(qubits[-1])
    for i in range(n):
        yield cirq.H(qubits[i])
    
    for x1, x2 in clauses:
        if x1 < 0:
            yield cirq.X(qubits[abs(x1) - 1])
        if x2 < 0:
            yield cirq.X(qubits[abs(x2) - 1])
        yield P1(qubits[-1]).controlled_by(qubits[abs(x1) - 1])
        yield P1(qubits[-1]).controlled_by(qubits[abs(x2) - 1])
        yield P2(qubits[-1]).controlled_by(qubits[abs(x1) - 1], qubits[abs(x2) - 1])

        if x1 < 0:
            yield cirq.X(qubits[abs(x1) - 1])
        if x2 < 0:
            yield cirq.X(qubits[abs(x2) - 1])
    
    for i in range(n):
        yield cirq.Rx(rads=2 * beta)(qubits[i])
    
    for i in range(n):
        yield cirq.measure(qubits[i])


def QAOA(n_lit, t, n_trials, clauses):
    simulator = Simulator()
    max_count = 0
    candidate = None
    for _ in range(n_trials):
        gamma, beta = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
        circuit = cirq.Circuit(qaoa_circuit(n_lit, gamma, beta, clauses))
        result = simulator.run(circuit)

        measurements = result.data.values.tolist()[0]
        count = 0
        for x1, x2 in clauses:
            sat = 0
            neg1, neg2 = x1 < 0, x2 < 0
            sat += (measurements[abs(x1) - 1] + neg1) % 2
            sat += (measurements[abs(x2) - 1] + neg2) % 2
            count += min(1, sat)
        if count > max_count:
            max_count = count
            candidate = measurements
    print("For clauses:")
    print(clauses)
    if max_count >= t:
        print("Found a satisfying literal string, " + "".join([str(i) for i in candidate]))
    else:
        print("After {} trials no satisfying literal string found.".format(n_trials))
    return max_count >= t