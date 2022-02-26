import cirq
import numpy as np
from typing import Callable, List, Optional, Sequence, Union
from collections import defaultdict
import time
from matplotlib import pyplot as plt

class helper:
    def bits_to_integer(self, bits):
        '''From a string of bits, return the integer representation.'''
        i = 0
        for b in bits[::-1]:
            i <<= 1
            i += b
        return int(i)

    def load_integer(self, qubits, integer):
        for q in qubits:
            if (integer % 2):
                yield  cirq.X(q)
            integer >>= 2

    def integer_to_bits(self, n, i):
        '''From integer, return string of bits representation. n is total number of bits.'''
        bits = []
        for _ in range(n):
            bits.append(i % 2)
            i >>= 1
        return bits

    def continued_fraction(self, p, q):
        '''Given p/q, return its continued fraction as a sequence.'''
        while q != 0:
            a = p // q
            yield a
            p, q = q, p - q * a

    def approximate_fraction(self, p, q, N):
        '''Given p/q, find the closest fraction a/b where b < N. Returns a tuple (a, b).'''
        ## truncate continued fraction expansion when denominator >= N
        a1, a2 = 1, 0
        b1, b2 = 0, 1
        truncated = False
        for k in self.continued_fraction(p, q):
            if k * b1 + b2 >= N:
                truncated = True
                break
            a1, a2 = k * a1 + a2, a1
            b1, b2 = k * b1 + b2, b1

        if truncated:
            ## use largest j where k/2 <= j < k and j*b1 + b2 < N.
            j = (N - b2) // b1
            if j >= k:
                pass
            elif k < 2 * j:  # found good j
                a1 = j * a1 + a2
                b1 = j * b1 + b2
            elif k == 2 * j:  # if k even, j = k/2 only admissible if the approximation is better
                next_a = j * a1 + a2
                next_b = j * b1 + b2
                if abs((p * 1.) / q - (next_a * 1.) / next_b) < abs((p * 1.) / q - (a1 * 1.) / b1):
                    a1, b1 = next_a, next_b
            ## else, no better approximation
        return a1, b1

class ModularExp(cirq.ArithmeticOperation):
    """Quantum modular exponentiation.
    This class represents the unitary which multiplies base raised to exponent
    into the target modulo the given modulus. More precisely, it represents the
    unitary V which computes modular exponentiation x**e mod n:
        V|y⟩|e⟩ = |y * x**e mod n⟩ |e⟩     0 <= y < n
        V|y⟩|e⟩ = |y⟩ |e⟩                  n <= y
    where y is the target register, e is the exponent register, x is the base
    and n is the modulus. Consequently,
        V|y⟩|e⟩ = (U**e|r⟩)|e⟩
    where U is the unitary defined as
        U|y⟩ = |y * x mod n⟩      0 <= y < n
        U|y⟩ = |y⟩                n <= y
    in the header of this file.
    Quantum order finding algorithm (which is the quantum part of the Shor's
    algorithm) uses quantum modular exponentiation together with the Quantum
    Phase Estimation to compute the order of x modulo n.
    """

    def __init__(
        self,
        target: Sequence[cirq.Qid],
        exponent: Union[int, Sequence[cirq.Qid]],
        base: int,
        modulus: int,
    ) -> None:
        if len(target) < modulus.bit_length():
            raise ValueError(
                f'Register with {len(target)} qubits is too small for modulus {modulus}'
            )
        self.target = target
        self.exponent = exponent
        self.base = base
        self.modulus = modulus

    def registers(self) -> Sequence[Union[int, Sequence[cirq.Qid]]]:
        return self.target, self.exponent, self.base, self.modulus

    def with_registers(
        self,
        *new_registers: Union[int, Sequence['cirq.Qid']],
    ) -> 'ModularExp':
        if len(new_registers) != 4:
            raise ValueError(
                f'Expected 4 registers (target, exponent, base, '
                f'modulus), but got {len(new_registers)}'
            )
        target, exponent, base, modulus = new_registers
        if not isinstance(target, Sequence):
            raise ValueError(f'Target must be a qubit register, got {type(target)}')
        if not isinstance(base, int):
            raise ValueError(f'Base must be a classical constant, got {type(base)}')
        if not isinstance(modulus, int):
            raise ValueError(f'Modulus must be a classical constant, got {type(modulus)}')
        return ModularExp(target, exponent, base, modulus)

    def apply(self, *register_values: int) -> int:
        assert len(register_values) == 4
        target, exponent, base, modulus = register_values
        if target >= modulus:
            return target
        return (target * base ** exponent) % modulus

    def _circuit_diagram_info_(
        self,
        args: cirq.CircuitDiagramInfoArgs,
    ) -> cirq.CircuitDiagramInfo:
        assert args.known_qubits is not None
        wire_symbols: List[str] = []
        t, e = 0, 0
        for qubit in args.known_qubits:
            if qubit in self.target:
                if t == 0:
                    if isinstance(self.exponent, Sequence):
                        e_str = 'e'
                    else:
                        e_str = str(self.exponent)
                    wire_symbols.append(f'ModularExp(t*{self.base}**{e_str} % {self.modulus})')
                else:
                    wire_symbols.append('t' + str(t))
                t += 1
            if isinstance(self.exponent, Sequence) and qubit in self.exponent:
                wire_symbols.append('e' + str(e))
                e += 1
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(wire_symbols))

class Order:
    def __init__(self, a, N, Threshold = 2):
        # In order to avoid wrong answers like 2*r or 3*r, double check.
        self.Threshold = Threshold
        self.a = a                          # Random number in (1, N)
        self.N = N                          # Target big integer
        self.n = int(np.log2(self.N)) + 1   # #qubits to represent N
        self.m = 2 * self.n                 # #qubits to represent order r
        self.table = defaultdict(int)
        # Prepare qubits for phase estimation
        k = cirq.GridQubit.rect(1, self.m, top=0)
        x = cirq.GridQubit.rect(1, self.n, top=1)
        ancillae = cirq.GridQubit.rect(1, self.m + 2, top=2)

        # Define operations for phase estimation on Ma
        self.ops = []
        self.ops.append(helper().load_integer(x, 1))
        self.ops.append(cirq.H(i) for i in k)
        self.ops.append(ModularExp(x, k+ancillae, self.a, self.N))
        self.ops.append(cirq.qft(*k[::-1], inverse=True))
        self.ops.append(cirq.measure(*k))

        # Build circuit for phase estimation on Ma
        self.circuit = cirq.Circuit(self.ops)

    def quantum_order_finder(self):
        while True:
            result = cirq.Simulator().run(self.circuit, repetitions=1)
            _, bits = result.measurements.popitem()
            k_over_r = helper().bits_to_integer(bits[0])
            # Uniformly draw from k/r where k=0, 1, ..., r-1.
            k, r = helper().approximate_fraction(k_over_r, 2 ** self.m, self.N)

            # Count potential order with a dictionary
            self.table[k] += 1

            # If this r has been observed {Threshold} times, it's very likely to be the order.
            if self.table[k] == self.Threshold and pow(self.a, k, self.N) == 1:
                return k

class Factorization():
    def __init__(self, N, k=40):
        self.N = N
        self.k = k

    def miller_rabin(self):
        # Returns True if n is a probable prime
        if self.N == 2 or self.N == 3:
            return True
        if self.N % 2 == 0:
            return False

        r, s = 0, self.N - 1
        while s % 2 == 0:
            r += 1
            s //= 2
        for _ in range(self.k):
            a = np.random.randint(2, self.N - 1)
            x = pow(a, s, self.N)
            if x == 1 or x == self.N - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, self.N)
                if x == self.N - 1:
                    break
            else:
                return False
        return True

    def interger_factorization(self):
        # Make sure integer N > 1
        try:
            assert isinstance(self.N, int) and self.N > 0
        except:
            print("Invalid input. Please enter an positive composite integer.")
        if self.N == 1:
            return 1

        # Make sure to deliver an even number to quantum simulator
        if self.N % 2 == 0 :            #
            self.N = self.N // 2
            return max(2, self.interger_factorization())

        # If N is a prime number, return 1. Check N's primality with Miller-Rabin primality test.
        if self.miller_rabin(): return self.N

        # Make sure N is not an integer power, othersise outputs the base.
        for exp in range(2, int(np.log(self.N)/np.log(3)) + 1):
            base = round(pow(self.N, 1./exp))
            if pow(base, exp) == self.N:
                return base

        # Call shor's algorithm
        return self.shor()

    def shor(self):
        # 1. Pick a random integer a, such that 1 < a < N
        a = np.random.randint(2, self.N)

        # 2. Compute greatest common divisor of a and N. If it's not 1, then we luckily find a non-trivial divisor.
        d = np.gcd(a, self.N)
        if d != 1:
            return d
        # 3. Call order-finding algorithm to compute a's order r modulo N
        r = Order(a, self.N).quantum_order_finder()

        # 4. If r is even and N | [a^(r/2) - 1] is false, then we find a non-trival divisor
        if r % 2 == 0:
            d = np.gcd(pow(a, r//2, self.N) - 1, self.N)
            if d != 1:
                return d
        # 5. Repeat the algorithm, until an factor is found
        return self.shor()

def test_factorization():
    Ns = [6, 15, 21]                            # Big prime number products
    n = [3, 4, 5]                               # #qubits to represent N
    use_time = []
    for N in Ns:
        start = time.time()
        factorizer = Factorization(N)
        print(factorizer.interger_factorization())

        how_long_it_took = time.time() - start
        use_time.append(how_long_it_took)
        how_long_it_took = "%.3f" %how_long_it_took
        print("Factorizing " + str(N) + " took " + how_long_it_took + " seconds.")

    plt.plot(use_time, n)
    plt.title("Scalability of Shor's algorithm")
    plt.xlabel('Bits of big integer')
    plt.ylabel('Runtime/s')
    plt.yscale('log')
    plt.show()

def test_order_finding(max_n):
    use_time = []
    n = np.arange(2, max_n + 1)                 # #bits to represent N
    for i in range(2, max_n + 1):               # from 2 to max_n bits
        N = 2 ** i - 1
        while True:
            a = np.random.randint(2, N)         # 1 < a < N
            if np.gcd(a, N) ==  1:              # gcd(a, N) == 1
                break
        start = time.time()
        order = Order(a, N)
        print("Quantum order finder found the order of " + str(a) + " modulo " + str(N))
        print(order.quantum_order_finder())
        how_long_it_took = time.time() - start
        use_time.append(how_long_it_took)
        how_long_it_took = "%.3f" %how_long_it_took
        print("It took " + how_long_it_took + " seconds.")

    plt.plot(use_time, n)
    plt.title("Scalability of Quantum Order Finder")
    plt.xlabel('Bits of big integer')
    plt.ylabel('Runtime/s')
    plt.yscale('log')
    plt.show()

def main():
    test_order_finding(4)

if __name__ == '__main__':
    main()


