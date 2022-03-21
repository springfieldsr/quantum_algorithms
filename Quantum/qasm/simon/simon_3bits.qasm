// Generated from Cirq v0.13.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
qreg q[6];
creg m_result[3];


h q[0];
h q[1];
h q[2];
cx q[0],q[3];
cx q[1],q[4];
cx q[2],q[5];
swap q[3],q[5];
h q[0];
h q[1];
h q[2];

// Gate: cirq.MeasurementGate(3, cirq.MeasurementKey(name='result'), ())
measure q[0] -> m_result[0];
measure q[1] -> m_result[1];
measure q[2] -> m_result[2];
