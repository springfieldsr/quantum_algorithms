import cirq

from utils import Oracle
from cirq import Simulator
from Deutsch_Jozsa import deutsch_jozsa_circuit

def bernstein_vazirani_circuit(f, n):
    return deutsch_jozsa_circuit(f, n)


def bernstein_vazirani_solver(f, n):
    circuit = bernstein_vazirani_circuit(f, n)
    simulator = Simulator()
    result = simulator.run(circuit)

    measurements = result.data.values.tolist()[0]
    a = "".join([str(i) for i in measurements])
    b = str(f("0"))

    return a, b