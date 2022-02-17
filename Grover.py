import random
import cirq
import numpy as np
from tqdm import tqdm
from cirq import X,Z,TOFFOLI
class Grover:
    def __init__(self, number_of_qubits, function):
        self.n = number_of_qubits
        theta = np.arcsin(1 / np.sqrt(2 ** self.n))
        k = np.pi / (4 * theta) - 1 / 2
        self.number_of_runs = int(np.floor(k))
        if np.sin((2*self.number_of_runs+1)*theta) < np.sin((2*self.number_of_runs+3)*theta):
            self.number_of_runs += 1

        self.f = function

    def grover_solver(self):
        # Set up input and output qubits.
        (input_qubits, output_qubit) = self.set_io_qubits()

        # Make oracle (black box)
        oracle = self.make_oracle(input_qubits, output_qubit)

        # Embed the oracle into a quantum circuit implementing Grover's algorithm.
        circuit = self.make_grover_circuit(input_qubits, output_qubit, oracle)

        # Sample from the circuit a couple times.
        simulator = cirq.Simulator()
        result = simulator.run(circuit,repetitions=100)
        frequencies = result.histogram(key='result', fold_func=self.bitstring)

        # Check if we actually found the secret value.
        most_common_bitstring = frequencies.most_common(1)[0][0]
        return most_common_bitstring

    def make_grover_circuit(self, input_qubits, output_qubit, oracle):
        """Find the value recognized by the oracle in sqrt(N) attempts."""
        # For 2 input qubits, that means using Grover operator only once.
        c = cirq.Circuit()
        c.append(
            [
                cirq.H(output_qubit),
                cirq.Z(output_qubit),
                cirq.H.on_each(*input_qubits),
            ]
        )

        for _ in range(self.number_of_runs):
            # Query oracle.
            c.append(self.make_oracle(input_qubits, output_qubit))

            # Construct Grover operator.
            c.append(cirq.H.on_each(*input_qubits))
            c.append(cirq.X.on_each(*input_qubits))
            cnX = X.controlled(self.n).on(*input_qubits[:self.n], output_qubit)
            c.append(cnX)
            c.append(cirq.X.on_each(*input_qubits))

            c.append(cirq.H.on_each(*input_qubits))
        # Measure the result.
        c.append(cirq.measure(*input_qubits, key='result'))
        return c

    def make_oracle(self, input_qubits, output_qubit):
        """Implement function {f(x) = 1 if x==x', f(x) = 0 if x!= x'}."""
        # Make oracle.
        # for (1, 1) it's just a Toffoli gate
        # otherwise negate the zero-bits.

        x_bits = [0] * self.n
        for i in range(2 ** self.n):
            if self.f(i) == 1:
                answer = ''.join(['0'] * (self.n - len(bin(i)[2:]))) + bin(i)[2:]
                for j in range(self.n):
                    x_bits[j] = int(answer[j])
                break
        cnX = X.controlled(self.n).on(*input_qubits[:self.n], output_qubit)

        yield (cirq.X(q) for (q, bit) in zip(input_qubits, x_bits) if not bit)
        yield cnX
        yield (cirq.X(q) for (q, bit) in zip(input_qubits, x_bits) if not bit)

    def set_io_qubits(self):
        """Add the specified number of input and output qubits."""
        input_qubits = [cirq.GridQubit(i, 0) for i in range(self.n)]
        output_qubit = cirq.GridQubit(self.n, 0)
        return (input_qubits, output_qubit)

    def bitstring(self, bits):
        return ''.join(str(int(b)) for b in bits)

