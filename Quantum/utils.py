import cirq
import numpy as np

class Oracle(cirq.Gate):
    def __init__(self, f, n, helper_bits):
        super(Oracle, self)
        self.f = f
        self.n = n
        self.helper_bits = helper_bits
    
    def _num_qubits_(self):
        return self.n + self.helper_bits
    
    def _decompose_(self, qubits):
        f = self.f
        n = self.n
        helper_bits = self.helper_bits

        for i in range(2**n):
            binary = "{0:b}".format(i)
            padding = "0" * (n - len(binary))
            binary = padding + binary

            for j in range(n):
                if binary[j] == '0':
                    yield cirq.X(qubits[j])

            output = f(binary)
            for j in range(helper_bits):
                if (int(output[j]) % 2 == 1):
                    if (helper_bits == 1):
                        yield cirq.optimizers.decompose_multi_controlled_x(list(qubits[:n]), qubits[n], list())
                    elif (j == 0):
                        yield cirq.optimizers.decompose_multi_controlled_x(list(qubits[:n]), qubits[n], list(qubits[n+1]))
                    else:
                        yield cirq.optimizers.decompose_multi_controlled_x(list(qubits[:n]), qubits[n + j], list(qubits[n]))
            
            for j in range(n):
                if binary[j] == '0':
                    yield cirq.X(qubits[j])

    def _circuit_diagram_info_(self):
        return "Uf"