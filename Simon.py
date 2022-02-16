import cirq
import numpy as np
import scipy as sp
from utils import Oracle
from cirq import Simulator
from collections import Counter

def make_oracle(input_qubits, output_qubits, secret_string):
    """Gates implementing the function f(a) = f(b) iff a ⨁ b = s"""
    # Copy contents to output qubits:
    for control_qubit, target_qubit in zip(input_qubits, output_qubits):
        yield cirq.CNOT(control_qubit, target_qubit)

    # Create mapping:
    if sum(secret_string):  # check if the secret string is non-zero
        # Find significant bit of secret string (first non-zero bit)
        significant = list(secret_string).index(1)

        # Add secret string to input according to the significant bit:
        for j in range(len(secret_string)):
            if secret_string[j] > 0:
                yield cirq.CNOT(input_qubits[significant], output_qubits[j])
    # Apply a random permutation:
    pos = [
        0,
        len(secret_string) - 1,
    ]  # Swap some qubits to define oracle. We choose first and last:
    yield cirq.SWAP(output_qubits[pos[0]], output_qubits[pos[1]])


def make_simon_circuit(input_qubits, output_qubits, oracle):
    """Solves for the secret period s of a 2-to-1 function such that
    f(x) = f(y) iff x ⨁ y = s
    """

    c = cirq.Circuit()

    # Initialize qubits.
    c.append(
        [
            cirq.H.on_each(*input_qubits),
        ]
    )

    # Query oracle.
    c.append(oracle)

    # Measure in X basis.
    c.append([cirq.H.on_each(*input_qubits), cirq.measure(*input_qubits, key='result')])

    return c


def post_processing(data, results):
    """Solves a system of equations with modulo 2 numbers"""
    sing_values = sp.linalg.svdvals(results)
    tolerance = 1e-5
    if sum(sing_values < tolerance) == 0:  # check if measurements are linearly dependent
        flag = True
        null_space = sp.linalg.null_space(results).T[0]
        solution = np.around(null_space, 3)  # chop very small values
        minval = abs(min(solution[np.nonzero(solution)], key=abs))
        solution = (solution / minval % 2).astype(int)  # renormalize vector mod 2
        data.append(''.join([str(x) for x in solution]))
        return flag
