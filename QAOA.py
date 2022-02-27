import cirq
import numpy as np
from cirq import Simulator


# Create the phaseshift gate
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
    qubits = cirq.LineQubit.range(n + 1)        # Add helper qubit
    P1 = PhaseShift(gamma)                      # Gate to shift up
    P2 = PhaseShift(-1 * gamma)                 # Gate to shift down

    yield cirq.X(qubits[-1])                    # Flip the helper bit to be 1
    for i in range(n):
        yield cirq.H(qubits[i])                 # Apply H gates to all first n qubits
    
    # Sep(gamma)
    for x1, x2 in clauses:
        if x1 + x2 == 0:                                                                # Special case when a clause contains one literal with opposite signs
            yield P1(qubits[-1])
            continue
        if x1 < 0:                                                                      # If literal is negative in current clause
            yield cirq.X(qubits[abs(x1) - 1])                                           # then negate it
        if x2 < 0:
            yield cirq.X(qubits[abs(x2) - 1])            
        yield P1(qubits[-1]).controlled_by(qubits[abs(x1) - 1])                         # Shift up if current state satisfies this clause
        yield P1(qubits[-1]).controlled_by(qubits[abs(x2) - 1])                         # Shift up if current state satisfies this clause

        # Shift down if both qubits satisfy this clause
        if abs(x1) != abs(x2):
            yield P2(qubits[-1]).controlled_by(qubits[abs(x1) - 1], qubits[abs(x2) - 1])
        else:
            yield P2(qubits[-1]).controlled_by(qubits[abs(x1) - 1])

        if x1 < 0:                              # Negate back if literals are negative
            yield cirq.X(qubits[abs(x1) - 1])
        if x2 < 0:
            yield cirq.X(qubits[abs(x2) - 1])
    
    # Mix(beta)
    for i in range(n):                          # Apply Rx(2 * beta) to all first n qubits
        yield cirq.Rx(rads=2 * beta)(qubits[i])
    
    for i in range(n):                          # Measure
        yield cirq.measure(qubits[i])


def QAOA(n_lit, t, n_trials, clauses, verbose=False):
    simulator = Simulator()
    max_count = 0
    candidate = None
    for _ in range(n_trials):
        gamma, beta = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)              # Randomly pick gamma and beta
        circuit = cirq.Circuit(qaoa_circuit(n_lit, gamma, beta, clauses))                       # Create circuit
        result = simulator.run(circuit)                                                         # Obtain results after meansurement

        measurements = result.data.values.tolist()[0]
        count = 0
        for x1, x2 in clauses:                                                                  # Calculate how many clauses current string satisfies
            sat = 0
            neg1, neg2 = x1 < 0, x2 < 0
            sat += (measurements[abs(x1) - 1] + neg1) % 2
            sat += (measurements[abs(x2) - 1] + neg2) % 2
            count += min(1, sat)
        if count > max_count:
            max_count = count
            candidate = measurements
            if max_count >= t: break
    if verbose:
        print("For clauses:")
        print(clauses)
        if max_count >= t:
            print("Found a satisfying literal string, " + "".join([str(i) for i in candidate]))
        else:
            print("After {} trials no satisfying literal string found.".format(n_trials))
    return max_count >= t