import cirq

from utils import Oracle
from cirq import Simulator
from Deutsch_Jozsa import deutsch_jozsa_circuit

def bernstein_vazirani_circuit(f, n):
    return deutsch_jozsa_circuit(f, n)


def bernstein_vazirani_solver(f, n):
    circuit = bernstein_vazirani_circuit(f, n)          # The structure of the quantum circuit is the same as in Deutsch-Josza
    simulator = Simulator()                             # Instantiate a simulator variable
    result = simulator.run(circuit)                     # Run the circuit on the simulator and save the result

    measurements = result.data.values.tolist()[0]       # Measure first n qubits and save them to a list, which is the "a" we are looking for
    a = "".join([str(i) for i in measurements])         # Save a as a string
    b = str(f("0"))                                     # Obtain b by calling f with input "0"

    return a, b