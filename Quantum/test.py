import os
import numpy as np
import cirq
from Grover import Grover


path = "qasm"
if not os.path.exists(path):
    os.mkdir(path)
