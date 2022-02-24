import numpy as np

class Order:
    def __init__(self, a, N):
        self.a = a
        self.N = N

    def quantum_order_finder(self):


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

        # Make sure to deliver an even number to quantum simulator
        if self.N % 2 == 0 :            #
            self.N = self.N // 2
            return self.interger_factorization()

        # If N is a prime number, return 1. Check N's primality with Miller-Rabin primality test.
        if self.miller_rabin(): return 1

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
            return max(a, d)

        # 3. Call order-finding algorithm to compute a's order r modulo N
        r = Order(a, N).quantum_order_finder()

        # 4. If r is even and N | [a^(r/2) - 1] is false, then we find a non-trival divisor
        if r % 2 == 0:
            d = np.gcd(pow(a, r//2, self.N) - 1, self.N)
            if d != 1:
                return d
        # 5. Repeat the algorithm, until an factor is found
        return self.shor()



def main(name):


if __name__ == '__main__':
    main()


