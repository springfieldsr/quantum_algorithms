// Generated from Cirq v0.13.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
qreg q[7];
creg m_result[5];


h q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
z q[5];
x q[0];
x q[1];
x q[2];
x q[4];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
ccx q[2],q[3],q[6];
cx q[4],q[0];
ry(pi*-0.25) q[3];
ry(pi*0.25) q[0];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
x q[2];
cx q[1],q[3];
h q[2];
ry(pi*-0.25) q[3];
x q[2];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
x q[1];
ry(pi*-0.25) q[0];
h q[1];
cx q[3],q[0];
x q[1];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
h q[3];
cx q[4],q[0];
x q[3];
ry(pi*0.25) q[0];
x q[4];
ccx q[2],q[3],q[6];
x q[0];
h q[4];
ry(pi*-0.25) q[3];
h q[0];
x q[4];
cx q[1],q[3];
x q[0];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
ccx q[2],q[3],q[6];
cx q[4],q[0];
ry(pi*-0.25) q[3];
ry(pi*0.25) q[0];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
x q[2];
cx q[1],q[3];
h q[2];
ry(pi*-0.25) q[3];
x q[2];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
x q[1];
ry(pi*-0.25) q[0];
h q[1];
cx q[3],q[0];
x q[1];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
x q[3];
cx q[4],q[0];
h q[3];
ry(pi*0.25) q[0];
x q[4];
ccx q[2],q[3],q[6];
x q[0];
h q[4];
ry(pi*-0.25) q[3];
h q[0];
x q[4];
cx q[1],q[3];
x q[0];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
ccx q[2],q[3],q[6];
cx q[4],q[0];
ry(pi*-0.25) q[3];
ry(pi*0.25) q[0];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
x q[2];
cx q[1],q[3];
h q[2];
ry(pi*-0.25) q[3];
x q[2];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
x q[1];
ry(pi*-0.25) q[0];
h q[1];
cx q[3],q[0];
x q[1];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
h q[3];
cx q[4],q[0];
x q[3];
ry(pi*0.25) q[0];
x q[4];
ccx q[2],q[3],q[6];
x q[0];
h q[4];
ry(pi*-0.25) q[3];
h q[0];
x q[4];
cx q[1],q[3];
x q[0];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
ccx q[2],q[3],q[6];
cx q[4],q[0];
ry(pi*-0.25) q[3];
ry(pi*0.25) q[0];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
x q[2];
cx q[1],q[3];
h q[2];
ry(pi*-0.25) q[3];
x q[2];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
x q[1];
ry(pi*-0.25) q[0];
h q[1];
cx q[3],q[0];
x q[1];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
x q[3];
cx q[4],q[0];
h q[3];
ry(pi*0.25) q[0];
x q[4];
ccx q[2],q[3],q[6];
x q[0];
h q[4];
ry(pi*-0.25) q[3];
h q[0];
x q[4];
cx q[1],q[3];
x q[0];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
ccx q[2],q[3],q[6];
cx q[4],q[0];
ry(pi*-0.25) q[3];
ry(pi*0.25) q[0];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
x q[2];
cx q[1],q[3];
h q[2];
ry(pi*-0.25) q[3];
x q[2];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
x q[1];
ry(pi*-0.25) q[0];
h q[1];
cx q[3],q[0];
x q[1];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
h q[3];
cx q[4],q[0];
x q[3];
ry(pi*0.25) q[0];
x q[4];
ccx q[2],q[3],q[6];
x q[0];
h q[4];
ry(pi*-0.25) q[3];
h q[0];
x q[4];
cx q[1],q[3];
x q[0];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
ccx q[2],q[3],q[6];
cx q[4],q[0];
ry(pi*-0.25) q[3];
ry(pi*0.25) q[0];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
x q[2];
cx q[1],q[3];
h q[2];
ry(pi*-0.25) q[3];
x q[2];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
x q[1];
ry(pi*-0.25) q[0];
h q[1];
cx q[3],q[0];
x q[1];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
x q[3];
cx q[4],q[0];
h q[3];
ry(pi*0.25) q[0];
x q[4];
ccx q[2],q[3],q[6];
x q[0];
h q[4];
ry(pi*-0.25) q[3];
h q[0];
x q[4];
cx q[1],q[3];
x q[0];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
ccx q[2],q[3],q[6];
cx q[4],q[0];
ry(pi*-0.25) q[3];
ry(pi*0.25) q[0];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
x q[2];
cx q[1],q[3];
h q[2];
ry(pi*-0.25) q[3];
x q[2];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
x q[1];
ry(pi*-0.25) q[0];
h q[1];
cx q[3],q[0];
x q[1];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
h q[3];
cx q[4],q[0];
x q[3];
ry(pi*0.25) q[0];
x q[4];
ccx q[2],q[3],q[6];
x q[0];
h q[4];
ry(pi*-0.25) q[3];
h q[0];
x q[4];
cx q[1],q[3];
x q[0];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
ccx q[2],q[3],q[6];
cx q[4],q[0];
ry(pi*-0.25) q[3];
ry(pi*0.25) q[0];
cx q[1],q[3];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
cx q[1],q[3];
ry(pi*0.25) q[3];
ccx q[2],q[3],q[6];
ry(pi*-0.25) q[3];
x q[2];
cx q[1],q[3];
h q[2];
ry(pi*-0.25) q[3];
cx q[0],q[3];
ry(pi*0.25) q[3];
ccx q[6],q[0],q[5];
cx q[1],q[3];
ry(pi*-0.25) q[0];
ry(pi*0.25) q[3];
cx q[4],q[0];
x q[1];
ry(pi*-0.25) q[0];
h q[1];
cx q[3],q[0];
ry(pi*0.25) q[0];
cx q[4],q[0];
ry(pi*0.25) q[0];
ccx q[6],q[0],q[5];
ry(pi*-0.25) q[0];
cx q[4],q[0];
ry(pi*-0.25) q[0];
cx q[3],q[0];
ry(pi*0.25) q[0];
x q[3];
cx q[4],q[0];
h q[3];
ry(pi*0.25) q[0];
x q[4];
x q[0];
h q[4];
h q[0];

// Gate: cirq.MeasurementGate(5, cirq.MeasurementKey(name='result'), ())
measure q[0] -> m_result[0];
measure q[1] -> m_result[1];
measure q[2] -> m_result[2];
measure q[3] -> m_result[3];
measure q[4] -> m_result[4];
