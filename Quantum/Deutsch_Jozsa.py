import cirq

from utils import Oracle
from cirq import Simulator

def deutsch_jozsa_circuit(f, n):
    qubits = cirq.LineQubit.range(n + 1)                        # Deutsch_Jozsa uses n+1 qubits
    # ancillae = cirq.LineQubit(n + 2)
    oracle = Oracle(f, n, 1)                                    # Initialize the oracle for solver, f as function, n as number of bits, 1 as number of
    m = n + 1                                                            # helper bits.
    def DJ_circuit():
        yield cirq.X(qubits[-1])                                # Reverse the last qubit to 1
        for i in range(n + 1):
            yield cirq.H(qubits[i])                             # Apply Hadamard to every qubit
        yield oracle.on(*qubits)                                # Apply Oracle(Bf) to every qubit
        for i in range(n):                                      # We only care about the first n qubits
            yield cirq.H(qubits[i])                             # Apply Hadamard to first n qubits, while ditch the helper qubit
        for i in range(n):                                      # Measure the first n qubits
            yield cirq.measure(qubits[i], key='q' + str(i))
    
    circuit = cirq.Circuit(DJ_circuit())                        # Establish the circuit and return it
    return circuit


def deutsch_jozsa_solver(f, n):
    circuit = deutsch_jozsa_circuit(f, n)                       # For the solver, we firstly build the circuit;
    simulator = Simulator()                                     # Then create a simulator variable;
    result = simulator.run(circuit)                             # Finally run the simulator and save the result

    measurements = result.data.values.tolist()[0]               # Transfer the result data to list, for the ease of testing.
    print(measurements)
    return 0 if sum(measurements) == 0 else 1                   # If all qubits are the same, they sum up either to n or 0, and we should output 0.
                                                                # Otherwise, f is a balanced function and we should outputs 1

def DJ_to_QASM(f, n):
    circuit = deutsch_jozsa_circuit(f, n)
    print(circuit.to_qasm())