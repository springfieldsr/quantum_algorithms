import cirq
import numpy as np
from cirq import Simulator
from tqdm import tqdm
import random


def qaoa_circuit(n, gamma, beta, clauses):
    qubits = cirq.LineQubit.range(n + 1)        # Add helper qubit

    yield cirq.X(qubits[-1])                    # Flip the helper bit to be 1
    for i in range(n):
        yield cirq.H(qubits[i])                 # Apply H gates to all first n qubits
    
    # Sep(gamma)
    for x1, x2 in clauses:
        if x1 + x2 == 0:                                                                # Special case when a clause contains one literal with opposite signs
            yield cirq.ZPowGate(exponent=-1 * gamma)(qubits[-1])
            continue
        if x1 < 0:                                                                      # If literal is negative in current clause
            yield cirq.X(qubits[abs(x1) - 1])                                           # then negate it
        if x2 < 0:
            yield cirq.X(qubits[abs(x2) - 1])            

        yield cirq.ZPowGate(exponent=-1 * gamma)(qubits[-1]).controlled_by(qubits[abs(x1) - 1])
        yield cirq.ZPowGate(exponent=-1 * gamma)(qubits[-1]).controlled_by(qubits[abs(x2) - 1]) 

        # Shift down if both qubits satisfy this clause
        if abs(x1) != abs(x2):
            yield cirq.ZPowGate(exponent=gamma)(qubits[-1]).controlled_by(qubits[abs(x1) - 1], qubits[abs(x2) - 1])
        else:
            yield cirq.ZPowGate(exponent=gamma)(qubits[-1]).controlled_by(qubits[abs(x1) - 1])

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


def QAOA_random_test(n_max_literals, number_of_tests):
    n_failed = 0
    for _ in range(number_of_tests):
        literals = list(range(1, n_max_literals + 1))
        n_clauses = random.randint(10, 30)                                  # Randomly choose clause length
        clauses = []
        literal_set = set()
        for _ in range(n_clauses):                                          # Randomly create list of clauses
            sign1, sign2 = random.choice([-1, 1]), random.choice([-1, 1])
            x1 = random.choice(literals) * sign1
            x2 = random.choice(literals) * sign2
            clauses.append((x1, x2))

            literal_set.add(abs(x1))
            literal_set.add(abs(x2))
        n_literals = len(literal_set)                                       # Get number of literals in clauses

        literals = list(literal_set)
        mapping = {literals[i]: i + 1 for i in range(n_literals)}
        for i in range(n_clauses):
            x1, x2 = clauses[i][0], clauses[i][1]
            clauses[i] = (mapping[abs(x1)] * abs(x1) // x1, mapping[abs(x2)] * abs(x2) // x2)
        t = 0
        for i in range(2 ** n_literals):
            binary = "{0:b}".format(i)
            padding = "0" * (n_literals - len(binary))
            binary = padding + binary
            string = []
            for j in range(n_literals):
                if binary[j] == '0':
                    string.append(-1 * literals[j])
                else:
                    string.append(literals[j])
            count = 0
            for x1, x2 in clauses:
                if x1 in string or x2 in string:
                    count += 1
            t = max(count ,t)
        try:
            assert QAOA(n_literals, t, 200, clauses)
        except:
            print("QAOA Test Failed. Returning the failed test case function...")
            print(clauses)
            n_failed += 1
    
    if n_failed == 0:
        print("QAOA Solver all clear for {} tests.".format(number_of_tests))
    else:
        print("QAOA solver failed {} times".format(n_failed))
    return n_failed


def QAOA_to_QASM(n_max_literals):
    literals = list(range(1, n_max_literals + 1))
    n_clauses = random.randint(10, 20)                                  # Randomly choose clause length
    clauses = []
    literal_set = set()
    for _ in range(n_clauses):                                          # Randomly create list of clauses
        sign1, sign2 = random.choice([-1, 1]), random.choice([-1, 1])
        x1 = random.choice(literals) * sign1
        x2 = random.choice(literals) * sign2
        clauses.append((x1, x2))

        literal_set.add(abs(x1))
        literal_set.add(abs(x2))
    n_literals = len(literal_set)                                       # Get number of literals in clauses

    literals = list(literal_set)
    mapping = {literals[i]: i + 1 for i in range(n_literals)}
    for i in range(n_clauses):
        x1, x2 = clauses[i][0], clauses[i][1]
        clauses[i] = (mapping[abs(x1)] * abs(x1) // x1, mapping[abs(x2)] * abs(x2) // x2)

    t = 0
    candidate = None
    for i in range(2 ** n_literals):
        binary = "{0:b}".format(i)
        padding = "0" * (n_literals - len(binary))
        binary = padding + binary
        string = []
        for j in range(n_literals):
            if binary[j] == '0':
                string.append(-1 * literals[j])
            else:
                string.append(literals[j])
        count = 0
        for x1, x2 in clauses:
            if x1 in string or x2 in string:
                count += 1
        if count > t:
            candidate = binary
            t = count
    gamma, beta = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)              # Randomly pick gamma and beta
    circuit = cirq.Circuit(qaoa_circuit(n_literals, gamma, beta, clauses))                       # Create circuit
    with open('QAOA.qasm', 'w') as f:
        f.write(circuit.to_qasm(header=str(clauses) + candidate + " " + str(t)))


def main():
    num_tests = 25

    print("QAOA Testing:")
    for n_bits in tqdm(range(5, 10)):
        QAOA_random_test(n_bits, num_tests)    

if __name__ == '__main__':
    main()