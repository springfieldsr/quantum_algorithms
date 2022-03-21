// Generated from Cirq v0.13.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
qreg q[8];
creg m_result[4];


h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[4];
cx q[1],q[5];
cx q[2],q[6];
cx q[3],q[7];
swap q[4],q[7];
h q[0];
h q[1];
h q[2];
h q[3];

// Gate: cirq.MeasurementGate(4, cirq.MeasurementKey(name='result'), ())
measure q[0] -> m_result[0];
measure q[1] -> m_result[1];
measure q[2] -> m_result[2];
measure q[3] -> m_result[3];
