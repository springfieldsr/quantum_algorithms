// Generated from Cirq v0.13.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [(0, 0), (1, 0)]
qreg q[2];
creg m_result[1];


h q[1];
h q[0];
z q[1];
x q[0];
cx q[0],q[1];
x q[0];
h q[0];
x q[0];
cx q[0],q[1];
x q[0];
h q[0];
measure q[0] -> m_result[0];
