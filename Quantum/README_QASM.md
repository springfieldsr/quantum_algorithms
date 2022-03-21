# Convert Cirq Circuits To QASM File
In this project, we setup methods to automatically convert Cirq cirquits to QASM files which is runnable on IBM quantum computers. The supportable algorithms are as following:  
* Deutsch Jozsa  
* Bernstein Vazirani  
* Simon  
* Grover  
* Shor  
* QAOA  
  
For each of these algorithms, a script is provided with easy to use methods for automatic conversion.  


* Simon's algorithm:
Due to some difficulty in decomposing original custom gates, I utilized a new oracle generator **Simon.py**, which easily parses into common gates in qasm files.
I wrote **Simon_Test** to generate qasm files and its folders.
```angular2html
python Simon_Test.py
```
* Shor's algorithm
Since we used a self-defined MExp gates, which is a *ControlledOperation* object and can not be decomposed by *to_qasm()*, we decided to test its scalability with cirq simulator by adding noise model to our original circuits. Let the circuit object calls **with_noise()** method and takes **cirq.depolarize()** as its input. We test with various parameter p to see how runtime is affected. This part is covered mainly by Yuchen Liu.
```angular2html
python Shor.py
```
* QAOA:  
```
from QAOA import QAOA_to_QASM
QAOA_to_QASM(n_qubits, n_files)  # specify the number of qubits (less than 5 as supported by IBM) and number of QASM files to generate
```
The above method generates **n_files** QASM files with distinct random gamma and beta.

* BV: Bernstein_Vazirani 
```
from Bernstein_Vazirani import BV_to_QASM
BV_to_QASM(n_qubits, f)  # specify n_qubits as number of qubits of problem, and f as the function to solve. It simply prints the cirquit that solves BV problem in qasm format.
```

* DJ: Deutsch_Jozsa
```
from Deutsch_Jozsa import DJ_to_QASM
DJ_to_QASM(n_qubits, f)  # specify n_qubits as number of qubits of problem, and f as the function to solve. It simply prints the cirquit that solves DJ problem in qasm format.
```


