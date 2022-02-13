import cirq
from Deutsch_Jozsa import deutsch_jozsa_solver
from Bernstein_Vazirani import bernstein_vazirani_solver
import numpy as np

def f(binary):
    return 0

print(deutsch_jozsa_solver(f, 2))
print(bernstein_vazirani_solver(f, 2))