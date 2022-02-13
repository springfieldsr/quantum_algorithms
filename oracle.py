import cirq
import numpy as np

def construct_oracle_matrix(f, n, helper_bits):
    m = n + helper_bits
    oracle = np.zeros((2**m ,2**m))

    for i in range(2**m):
        binary = bin(i)
        output = f(binary[:-1])
        if output == 1:
            binary[-1] = str(1 - int(binary[-1]))
        oracle[i][int(binary, 2)] = 1
    
    return oracle

class Oracle(cirq.Gate):
    def __init__(self, f, n, helper_bits):
        super(Oracle, self)
        self.f = f
        self.n = n
        self.helper_bits = helper_bits
    
    def _num_qubits_(self):
        return self.n + self.helper_bits

    def _unitary_(self):
        return construct_oracle_matrix(self.f, self.n, self.helper_bits)

    def _circuit_diagram_info_(self):
        return "Uf"