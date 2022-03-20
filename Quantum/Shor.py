import cirq
import numpy as np
from typing import Callable, List, Optional, Sequence, Union
from collections import defaultdict
import time
from matplotlib import pyplot as plt

class helper:
    # Some functions that help communicate between integer and qubits.
    def bits_to_integer(self, bits):
        # Translate a list of bits into an integer
        i = 0
        for b in bits[::-1]:
            i <<= 1
            i += b
        return int(i)

    def load_integer(self, qubits, integer):
        # load a large integer to qubits in binary
        for q in qubits:
            if (integer % 2):
                yield  cirq.X(q)
            integer >>= 2

    def integer_to_bits(self, n, i):
        # Translate integer i a list of binary bits.
        bits = []
        for _ in range(n):
            bits.append(i % 2)
            i >>= 1
        return bits

    def continued_fraction(self, p, q):
        # Given to integer p and q, return floor(p/q) as a sequence
        # for the purpose of calculating order.
        while q != 0:
            a = p // q
            yield a
            p, q = q, p - q * a

    def approximate_fraction(self, p, q, N):
        # Given p/q, find the closest fraction a/b where b < N. Returns a tuple (a, b).'''
        # Truncate continued fraction expansion when denominator >= N
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


# Here we borrow 4 Quantum gates for arithmetic operation from
# https://github.com/kevinddchen/Cirq-PrimeFactorization/blob/main/arithmetic.py
# The implementation is based on https://arxiv.org/abs/quant-ph/9511018.

class Add(cirq.Gate):
    '''Add classical integer a to qubit b. To account for possible overflow, an
    extra qubit (initialized to zero) must be supplied for b. Uses O(n) elementary
    gates.

      |b> --> |b+a>

    Parameters:
      n: number of qubits.
      a: integer, 0 <= a < 2^n.

    Input to gate is 2n+1 qubits split into:
      n+1 qubits for b, 0 <= b < 2^n. The most significant digit is initialized to 0. b+a is saved here.
      n ancillary qubits initialized to 0. Unchanged by operation.
    '''

    def __init__(self, n, a):
        super().__init__()
        self.n = n
        self.a = a

    def _num_qubits_(self):
        return 2 * self.n + 1

    def _circuit_diagram_info_(self, args):
        return ["Add_b"] * (self.n + 1) + ["Add_anc"] * self.n

    def _decompose_(self, qubits):
        n = self.n
        a = helper().integer_to_bits(n, self.a)
        b = qubits[:n]
        anc = qubits[n + 1:] + (qubits[n],)  # internally, b[n] is placed in anc[n]

        ## In forward pass, store carried bits in ancilla.
        for i in range(n):
            if a[i]:
                yield cirq.CNOT(b[i], anc[i + 1])
                yield cirq.X(b[i])
            yield cirq.TOFFOLI(anc[i], b[i], anc[i + 1])
        yield cirq.CNOT(anc[n - 1], b[n - 1])
        ## In backward pass, undo carries, then add a and carries to b.
        for i in range(n - 2, -1, -1):
            yield cirq.TOFFOLI(anc[i], b[i], anc[i + 1])
            if a[i]:
                yield cirq.X(b[i])
                yield cirq.CNOT(b[i], anc[i + 1])
                yield cirq.X(b[i])
            yield cirq.CNOT(anc[i], b[i])


class MAdd(cirq.Gate):
    '''Add classical integer a to qubit b, modulo N. Integers a and b must be less
    than N for correct behavior. Uses O(n) elementary gates.

      |b> --> |b+a mod N>

    Parameters:
      n: number of qubits.
      a: integer, 0 <= a < N.
      N: integer, 1 < N < 2^n.

    Input to gate is 2n+2 qubits split into:
      n qubits for b, 0 <= b < N. a+b mod N is saved here.
      n+2 ancillary qubits initialized to 0. Unchanged by operation.
    '''

    def __init__(self, n, a, N):
        super().__init__()
        self.n = n
        self.a = a
        self.N = N

    def _num_qubits_(self):
        return 2 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["MAdd_b"] * self.n + ["MAdd_anc"] * (self.n + 2)

    def _decompose_(self, qubits):
        n = self.n
        b = qubits[:n + 1]  # extra qubit for overflow
        anc = qubits[n + 1:2 * n + 1]
        t = qubits[2 * n + 1]

        Add_a = Add(n, self.a)
        Add_N = Add(n, self.N)
        yield Add_a.on(*b, *anc)
        yield cirq.inverse(Add_N).on(*b, *anc)
        ## Second register is a+b-N. The most significant digit indicates underflow from subtraction.
        yield cirq.CNOT(b[n], t)
        yield Add_N.controlled(1).on(t, *b, *anc)
        ## To reset t, subtract a from second register. If underflow again, means that t=0 previously.
        yield cirq.inverse(Add_a).on(*b, *anc)
        yield cirq.X(b[n])
        yield cirq.CNOT(b[n], t)
        yield cirq.X(b[n])
        yield Add_a.on(*b, *anc)


class MMult(cirq.Gate):
    '''Multiply qubit x by classical integer a, modulo N. Exact map is:

      |x; b> --> |x; b + x*a mod N>

    Integers a, b, and x must be less than N for correct behavior. Uses O(n^2)
    elementary gates.

    Parameters:
      n: number of qubits.
      a: integer, 0 <= a < N.
      N: integer, 1 < N < 2^n.

    Input to gate is 3n+2 qubits split into:
      n qubits for x, 0 <= x < N. Unchanged by operation.
      n qubits for b, 0 <= b < N. b + x*a mod N is saved here.
      n+2 ancillary qubits initialized to 0. Unchanged by operation.
    '''

    def __init__(self, n, a, N):
        super().__init__()
        self.n = n
        self.a = a
        self.N = N

    def _num_qubits_(self):
        return 3 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["MMult_x"] * self.n + ["MMult_b"] * self.n + ["MMult_anc"] * (self.n + 2)

    def _decompose_(self, qubits):
        n = self.n
        N = self.N
        x = qubits[:n]
        b = qubits[n:2 * n]
        anc = qubits[2 * n:]

        ## x*a = 2^(n-1) x_(n-1) a + ... + 2 x_1 a + x_0 a
        ## so the bits of x control the addition of a * 2^i
        d = self.a  # stores a * 2^i mod N
        for i in range(n):
            yield MAdd(n, d, N).controlled(1).on(x[i], *b, *anc)
            d = (d << 1) % N


class Ua(cirq.Gate):
    '''Applies the unitary n-qubit operation,

      |x> --> |x*a mod N>

    where gcd(a, N) = 1. Integers a and x must be less than N for correct
    behavior. Uses O(n^2) elementary gates.

    Parameters:
      n: number of qubits.
      a: integer, 0 < a < N and gcd(a, N) = 1.
      N: integer, 1 < N < 2^n.
      inv_a: (optional) integer, inverse of a mod N. Skips recalculation of this if provided.

    Input to gate is 3n+2 qubits split into:
      n qubits for x, 0 <= x < N. x*a mod N is saved here.
      2n+2 ancillary qubits initialized to 0. Unchanged by operation.
    '''

    def __init__(self, n, a, N, inv_a=None):
        super().__init__()
        self.n = n
        self.a = a
        self.N = N
        if inv_a:
            self.inv_a = inv_a
        else:
            self.inv_a = pow(a, -1, N)

    def _num_qubits_(self):
        return 3 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["Ua_x"] * self.n + ["Ua_anc"] * (2 * self.n + 2)

    def _decompose_(self, qubits):
        n = self.n
        N = self.N
        x = qubits[:n]
        anc_mult = qubits[n:2 * n]
        anc_add = qubits[2 * n:]

        yield MMult(n, self.a, N).on(*x, *anc_mult, *anc_add)
        for i in range(n):
            yield cirq.SWAP(x[i], anc_mult[i])
        yield cirq.inverse(MMult(n, self.inv_a, N)).on(*x, *anc_mult, *anc_add)


class MExp(cirq.Gate):
    '''Multiply qubit x by a^k, modulo N, where a is a classical integer and
    gcd(a, N) = 1. Integers a and x must be less than N for correct behavior. Uses
    O(m * n^2) elementary gates.

      |k; x> --> |k; x * a^k mod N>

    Parameters:
      m: number of qubits for k.
      n: number of qubits for x.
      a: integer, 0 < a < N and gcd(a, N) = 1.
      N: integer, 1 < N < 2^n.

    Input to gate is m+3n+2 qubits split into:
      m qubits for k, 0 <= k < 2^m. Unchanged by operation.
      n qubits for x, 0 <= x < N. x * a^k mod N is saved here.
      2n+2 ancillary qubits initialized to 0. Unchanged by operation.
    '''

    def __init__(self, m, n, a, N):
        super().__init__()
        self.m = m
        self.n = n
        self.a = a
        self.inv_a = pow(a, -1, N)  # inverse of a mod N
        self.N = N

    def _num_qubits_(self):
        return self.m + 3 * self.n + 2

    def _circuit_diagram_info_(self, args):
        return ["MExp_k"] * self.m + ["MExp_x"] * self.n + ["MExp_anc"] * (2 * self.n + 2)

    def _decompose_(self, qubits):
        m = self.m
        n = self.n
        N = self.N
        k = qubits[:m]
        x = qubits[m:m + n]
        anc = qubits[m + n:]

        d = self.a  # stores a^(2^i)
        inv_d = self.inv_a  # stores a^(-2^i)
        for i in range(m):
            yield Ua(n, d, N, inv_d).controlled(1).on(k[i], *x, *anc)
            d = (d * d) % N
            inv_d = (inv_d * inv_d) % N

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
        k = cirq.GridQubit.rect(1, self.m, top=0)                                   # a^k = a^k0 a^2k1 a^4k2 ... a^(2^m-1)km-1
        x = cirq.GridQubit.rect(1, self.n, top=1)                                   # x = 0, 1, ..., N - 1
        ancillae = cirq.GridQubit.rect(1, 2*self.n + 2, top=2)                      # ancillary qubits for keeping unitary

        # Define operations for phase estimation on Ma
        self.ops = []
        self.ops.append(helper().load_integer(x, 1))                                # Load 1 to x
        self.ops.append(cirq.H(i) for i in k)                                       # Apply Hadamard to k
        self.ops.append(MExp(self.m, self.n, self.a, self.N).on(*k, *x, *ancillae)) # Calculate modular exponentiation
        self.ops.append(cirq.qft(*k[::-1], inverse=True))                           # Apply quantum Fourier transform
        self.ops.append(cirq.measure(*k))                                           # Measure k as output
        # Build circuit for phase estimation on Ma
        self.circuit = cirq.Circuit(self.ops)

    def quantum_order_finder(self):
        while True:
            result = cirq.Simulator().run(self.circuit, repetitions=1)              # Measure k as k/r
            _, bits = result.measurements.popitem()
            k_over_r = helper().bits_to_integer(bits[0])                            # Read k/r as integer
            # Uniformly draw from k/r where k=0, 1, ..., r-1.
            k, r = helper().approximate_fraction(k_over_r, 2 ** self.m, self.N)     # Obtain order r via approximate fraction

            # Count potential order with a dictionary, precluiding of getting 2*r, which is improbable though.
            self.table[r] += 1
            print(self.table)
            # If this r has been observed {Threshold} times, and r is the order we are looking for, output it.
            if self.table[r] == self.Threshold and pow(self.a, r, self.N) == 1:
                return r

class Factorization():
    def __init__(self, N, k=40):
        self.N = N
        self.k = k

    def miller_rabin(self):
        # Adapted the Miller Rabin Algorithm to check primality from
        # https://gist.github.com/Ayrx/5884790

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
        # Handle the input 1, both as original input and recursive input.
        if self.N == 1:
            return 1

        # Make sure to deliver an even number to quantum simulator
        if self.N % 2 == 0 :            #
            self.N = self.N // 2
            return max(2, self.interger_factorization())

        # If N is a prime number, return 1. Check N's primality with Miller-Rabin primality test.
        if self.miller_rabin(): return self.N

        # Make sure N is not an integer power, otherwise outputs the base.
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

        # 4. If r is even and N | [a^(r/2) - 1] is false, then try to find a non-trivial divisor
        if r % 2 == 0:
            d = np.gcd(pow(a, r//2, self.N) - 1, self.N)
            if d != 1:
                return d
        # 5. Repeat the algorithm, until an factor is found
        return self.shor()

"""def test_factorization():
    Ns = [6, 15, 21]                            # Big prime number products
    n = [3, 4, 5]                               # #qubits to represent N
    use_time = []
    for N in Ns:
        start = time.time()
        factorizer = Factorization(N)           # Instantiate the circuit
        print(factorizer.interger_factorization())

        how_long_it_took = time.time() - start  # Calculate runtime
        use_time.append(how_long_it_took)
        how_long_it_took = "%.3f" %how_long_it_took
        print("Factorizing " + str(N) + " took " + how_long_it_took + " seconds.")

    plt.plot(n, use_time)
    plt.title("Scalability of Shor's algorithm")
    plt.xlabel('Bits of big integer')
    plt.ylabel('Runtime/s')
    plt.yscale('log')
    plt.show()

def test_order_finding(max_n):
    use_time = []
    n = []                                      # #bits to represent N
    for i in range(2, max_n + 1):               # from 2 to max_n bits
        N = 2 ** i - 1                          # N is the all-one i bits binary number
        while True:                             # Randomly generate a, such that
            a = np.random.randint(2, N)         # 1 < a < N
            if np.gcd(a, N) ==  1:              # and gcd(a, N) == 1
                break
        start = time.time()
        order = Order(a, N)
        try:                                    # For the purpose of automatic test, allow for any max_n. Thus, it may run out of RAM
            print(order.quantum_order_finder())
            print("Quantum order finder found the order of " + str(a) + " modulo " + str(N))
            how_long_it_took = time.time() - start
            use_time.append(how_long_it_took)
            n.append(i)
            how_long_it_took = "%.3f" %how_long_it_took
            print("It took " + how_long_it_took + " seconds.")
        except:
            print("When finding order of a "+ str(i) + " bits number, it runs out of RAM!")
            break
    plt.plot(n, use_time)
    plt.title("Scalability of Quantum Order Finder")
    plt.xlabel('Bits of big integer')
    plt.ylabel('Runtime/s')
    plt.yscale('log')
    plt.show()

def main():
    # Test quantum order finder, one input is the maximum bits to represent N. a is randomly generated from
    # (1, N). Test function can automatically deal with crash due to running out of memory.
    test_order_finding(7)

    # Test the entire integer factorization algorithm. We pick 3 integers that are worth of factorization inside
    # the function. It can take a really long time to factorize even a small integer, but sometimes it's a lot faster
    # when the classical algorithm luckily picks its factor.
    test_factorization()

if __name__ == '__main__':
    main()
"""


