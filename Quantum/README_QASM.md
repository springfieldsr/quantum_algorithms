# Convert Cirq Circuits To QASM File
In this project, we setup methods to automatically convert Cirq cirquits to QASM files which is runnable on IBM quantum computers. The supportable algorithms are as following:  
* Deutsch Jozsa  
* Bernstein Vazirani  
* Simon  
* Grover  
* Shor  
* QAOA  
  
For each of these algorithms, a script is provided with easy to use methods for automatic conversion.  
* QAOA:  
```
from QAOA import QAOA_to_QASM
QAOA_to_QASM(n_qubits, n_files)  # specify the number of qubits (less than 5 as supported by IBM) and number of QASM files to generate
```
The above method generates **n_files** QASM files with distinct random gamma and beta.