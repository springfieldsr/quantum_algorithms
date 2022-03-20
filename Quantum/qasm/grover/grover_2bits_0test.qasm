// Generated from Cirq v0.13.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [(0, 0), (1, 0), (2, 0)]
qreg q[3];
creg m_result[2];


h q[2];
h q[0];
h q[1];
z q[2];
x q[0];
ccx q[0],q[1],q[2];
x q[0];
h q[1];
h q[0];
x q[1];
x q[0];
ccx q[0],q[1],q[2];
x q[0];
x q[1];
h q[0];
h q[1];

// Gate: cirq.MeasurementGate(2, cirq.MeasurementKey(name='result'), ())
measure q[0] -> m_result[0];
measure q[1] -> m_result[1];
